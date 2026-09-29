# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""Extract a Qwen3.5 MTP (multi-token prediction) draft head into an MTP-only GGUF.

Sources may be full checkpoints (GGUF with llama.cpp ``blk.N.nextn.*`` MTP
tensors, or Hugging Face safetensors with ``mtp.*`` keys) or MTP-only files in
any naming convention. The output is named with HF-canonical ``mtp.*`` keys
and loads through the CLIPLoader (GGUF + MTP) node.

Usage:
    python tools/extract_mtp.py SRC [SRC ...] --out 9b-mtp-q8.gguf [--type Q8_0]

Supported --type values are F16 and the gguf-py quantization types (Q4_0,
Q8_0, Q4_K, ...). 1D tensors (norms) are always kept in F16.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import gguf
import numpy as np
import torch

REPO = Path(__file__).parents[1]

_spec = importlib.util.spec_from_file_location(
    "comfyui_gguf_extract.loader", REPO / "loader.py",
    submodule_search_locations=[str(REPO)],
)
loader = importlib.util.module_from_spec(_spec)
sys.modules["comfyui_gguf_extract"] = loader
sys.modules["comfyui_gguf_extract.loader"] = loader
_spec.loader.exec_module(loader)


def load_safetensors(path):
    from safetensors.torch import load_file
    return load_file(str(path))


def looks_like_mtp(name):
    return name.startswith("mtp") or ".mtp." in name or "nextn" in name


def collect_mtp_state_dict(paths):
    sd = {}
    for path in paths:
        path = Path(path)
        if path.suffix == ".gguf":
            part = loader.gguf_mtp_loader(str(path))
        elif path.suffix == ".safetensors":
            raw = {k: v for k, v in load_safetensors(path).items() if looks_like_mtp(k)}
            if not raw:
                raise ValueError(f"No MTP tensors found in {path}")
            part = loader.normalize_mtp_keys(raw)
        else:
            raise ValueError(f"Unsupported source type: {path}")
        overlap = set(sd) & set(part)
        if overlap:
            print(f"note: {path} overrides {sorted(overlap)}")
        sd.update(part)
    return sd


def quantize_tensor(name, tensor, qtype):
    # GGUF sources may already be packed quant blocks; dequantize to values first
    # so re-quantization never eats raw block bytes as if they were weights.
    if getattr(tensor, "tensor_type", None) not in (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        tensor = loader.dequantize_tensor(tensor, torch.float32)
    array = tensor.to(torch.float32).numpy() if tensor.dtype != torch.float32 else tensor.numpy()
    if qtype is None or qtype == gguf.GGMLQuantizationType.F16 or array.ndim < 2:
        return array.astype(np.float16), gguf.GGMLQuantizationType.F16
    block_size, _ = gguf.GGML_QUANT_SIZES[qtype]
    if array.shape[-1] % block_size:
        print(f"warning: {name} width {array.shape[-1]} is not a multiple of {block_size}; keeping F16")
        return array.astype(np.float16), gguf.GGMLQuantizationType.F16
    try:
        return np.asarray(gguf.quants.quantize(array, qtype)), qtype
    except Exception as exc:
        print(f"warning: cannot quantize {name} to {qtype.name} ({exc}); keeping F16")
        return array.astype(np.float16), gguf.GGMLQuantizationType.F16


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", help="checkpoint files containing MTP weights (.gguf / .safetensors)")
    parser.add_argument("--out", required=True, help="output MTP-only GGUF path")
    parser.add_argument("--type", default="Q8_0", help="quantization type for 2D tensors (default Q8_0; F16 keeps everything)")
    args = parser.parse_args()

    qtype = None if args.type.upper() == "F16" else gguf.GGMLQuantizationType[args.type.upper()]
    sd = collect_mtp_state_dict(args.sources)

    if "mtp.fc.weight" not in sd:
        raise SystemExit(f"sources do not contain a complete MTP head (missing mtp.fc.weight): {args.sources}")
    missing = [name for name in (
        "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight",
        "mtp.layers.0.input_layernorm.weight", "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_proj.weight", "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight", "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight", "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.mlp.down_proj.weight",
    ) if name not in sd]
    if missing:
        print(f"warning: incomplete MTP head, missing: {missing}")

    write_mtp_gguf(sd, args.out, qtype)


def write_mtp_gguf(sd, out_path, qtype):
    """Write an MTP state dict as GGUF with logical dims preserved."""
    writer = gguf.GGUFWriter(str(out_path), arch="qwen35")
    writer.add_name(Path(out_path).stem)
    for name in sorted(sd):
        tensor = sd[name]
        logical_shape = tuple(int(v) for v in getattr(tensor, "tensor_shape", tensor.shape))
        array, written_type = quantize_tensor(name, tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor, qtype)
        # quantized arrays are packed (rows, bytes_per_row); GGUF dims must
        # stay logical - mirror tools/convert.py's comfy.gguf.orig_shape contract
        writer.add_array(f"comfy.gguf.orig_shape.{name}", logical_shape)
        writer.add_tensor(name, array, raw_dtype=written_type)
        print(f"  {name:48s} {logical_shape} -> {written_type.name}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {out_path} ({len(sd)} tensors)")


if __name__ == "__main__":
    main()
