# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import warnings
import logging
import torch
import gguf
import json
import re
import os
import threading
import comfy.memory_management
from .ops import GGMLLayer, GGMLTensor
from .dequant import apply_tensor_postprocess, is_quantized, dequantize_tensor, triton_in_capture_usable, warm_device_constants
from .quant_matmul import k_quant_matmul_available, unpin_host_weight
from .quant_ops import make_quantized

IMG_ARCH_LIST = {"flux", "sd1", "sdxl", "sd3", "aura", "hidream", "cosmos", "ltxv", "hyvid", "wan", "lumina2", "qwen_image", "ideogram", "krea2", "minimax_h3"}
TXT_ARCH_LIST = {"t5", "t5encoder", "llama", "qwen2vl", "qwen3", "qwen3vl", "qwen35", "gemma3"}
VIS_TYPE_LIST = {"clip-vision", "mmproj"}

def device_supports_bf16():
    """
    Return True if the active torch device can run bf16 natively. On devices
    without native bf16 support, computation silently falls back to fp32 which
    is very slow, so callers should load tensors as fp16 instead.
    """
    try:
        import comfy.model_management
        return comfy.model_management.should_use_bf16(comfy.model_management.get_torch_device())
    except Exception:
        # If support can't be determined, keep the previous bf16 behavior.
        return True


def dynamic_gguf_file_slice(path):
    """
    Create the file handle metadata used by DynamicVRAM to transfer a GGUF
    tensor directly from its mmap-backed file to GPU memory.
    """
    if not comfy.memory_management.aimdo_enabled:
        return None

    import comfy_aimdo.model_mmap

    model_mmap = comfy_aimdo.model_mmap.ModelMMAP(path)
    return model_mmap, threading.Lock()


def attach_dynamic_file_slice(torch_tensor, model_mmap, file_lock, offset, size):
    storage = torch_tensor.untyped_storage()
    storage._comfy_tensor_file_slice = comfy.memory_management.TensorFileSlice(
        model_mmap.get_file_handle(),
        file_lock,
        offset,
        size,
    )
    # Keep the native mmap alive for the storage lifetime.
    storage._comfy_tensor_mmap_refs = (model_mmap,)

def get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))

def get_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        # extra check here as this is used for checking arch string
        if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
            raise TypeError(f"Bad type for GGUF {field_name} key: expected string, got {field.types!r}")
        return str(field.parts[field.data[-1]], encoding="utf-8")
    elif field_type in [int, float, bool]:
        return field_type(field.parts[field.data[-1]].item())
    else:
        raise TypeError(f"Unknown field type {field_type}")

def get_list_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        return tuple(str(field.parts[part_idx], encoding="utf-8") for part_idx in field.data)
    elif field_type in [int, float, bool]:
        return tuple(field_type(field.parts[part_idx][0]) for part_idx in field.data)
    else:
        raise TypeError(f"Unknown field type {field_type}")

def get_gguf_metadata(reader):
    """Extract all simple metadata fields like safetensors"""
    metadata = {}
    for field_name in reader.fields:
        try:
            field = reader.get_field(field_name)
            if len(field.types) == 1:  # Simple scalar fields only
                if field.types[0] == gguf.GGUFValueType.STRING:
                    metadata[field_name] = str(field.parts[field.data[-1]], "utf-8")
                elif field.types[0] == gguf.GGUFValueType.INT32:
                    metadata[field_name] = int(field.parts[field.data[-1]])
                elif field.types[0] == gguf.GGUFValueType.F32:
                    metadata[field_name] = float(field.parts[field.data[-1]])
                elif field.types[0] == gguf.GGUFValueType.BOOL:
                    metadata[field_name] = bool(field.parts[field.data[-1]])
        except:
            continue
    return metadata

def gguf_tensor_count(path):
    return len(gguf.GGUFReader(path).tensors)


def gguf_sd_loader(path, handle_prefix="model.diffusion_model.", is_text_model=False, dynamic=False, progress_callback=None):
    """
    Read state dict as fake tensors
    """
    reader = gguf.GGUFReader(path)
    dynamic_file_slice = dynamic_gguf_file_slice(path) if dynamic else None

    # filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    # detect and verify architecture
    compat = None
    arch_str = get_field(reader, "general.architecture", str)
    type_str = get_field(reader, "general.type", str)
    if arch_str in [None, "pig", "cow"]:
        if is_text_model:
            raise ValueError(f"This gguf file is incompatible with llama.cpp!\nConsider using safetensors or a compatible gguf file\n({path})")
        compat = "sd.cpp" if arch_str is None else arch_str
        # import here to avoid changes to convert.py breaking regular models
        from .tools.convert import detect_arch
        try:
            arch_str = detect_arch(set(val[0] for val in tensors)).arch
        except Exception as e:
            raise ValueError(f"This model is not currently supported - ({e})")
    elif arch_str not in TXT_ARCH_LIST and is_text_model:
        if type_str not in VIS_TYPE_LIST:
            raise ValueError(f"Unexpected text model architecture type in GGUF file: {arch_str!r}")
    elif arch_str not in IMG_ARCH_LIST and not is_text_model:
        raise ValueError(f"Unexpected architecture type in GGUF file: {arch_str!r}")

    qwen35_postprocess = {}
    if arch_str == "qwen35":
        num_k_heads = get_field(reader, "qwen35.ssm.group_count", int)
        num_v_heads = get_field(reader, "qwen35.ssm.time_step_rank", int)
        key_head_dim = get_field(reader, "qwen35.ssm.state_size", int)
        inner_size = get_field(reader, "qwen35.ssm.inner_size", int)
        if not all((num_k_heads, num_v_heads, key_head_dim, inner_size)):
            raise ValueError("Qwen3.5 GGUF is missing required linear-attention metadata")
        if num_v_heads % num_k_heads or inner_size % num_v_heads:
            raise ValueError("Qwen3.5 GGUF has inconsistent linear-attention metadata")
        num_v_per_k = num_v_heads // num_k_heads
        value_head_dim = inner_size // num_v_heads
        if num_v_per_k > 1:
            transform = lambda dim, start, head_dim: (
                "qwen35_inverse_v_heads", dim, start,
                num_k_heads, num_v_per_k, head_dim,
            )
            qk_size = num_k_heads * key_head_dim * 2
            for block in range(get_field(reader, "qwen35.block_count", int)):
                prefix = f"blk.{block}."
                qwen35_postprocess.update({
                    prefix + "attn_qkv.weight": (transform(0, qk_size, value_head_dim),),
                    prefix + "attn_gate.weight": (transform(0, 0, value_head_dim),),
                    prefix + "ssm_alpha.weight": (transform(0, 0, 1),),
                    prefix + "ssm_beta.weight": (transform(0, 0, 1),),
                    prefix + "ssm_conv1d.weight": (transform(0, qk_size, value_head_dim),),
                    prefix + "ssm_dt.bias": (transform(0, 0, 1),),
                    prefix + "ssm_a": (transform(0, 0, 1),),
                    prefix + "ssm_out.weight": (transform(1, 0, value_head_dim),),
                })

    if compat:
        logging.warning(f"Warning: This gguf model file is loaded in compatibility mode '{compat}' [arch:{arch_str}]")

    # Q8_CR weights must use ComfyUI's native TensorWiseINT8Layout rather than
    # the generic GGML layout so DynamicVRAM retains native INT8 ConvRot kernels.
    custom_quant_configs = {}
    for field_name in reader.fields:
        if field_name.startswith("comfy.gguf.quant."):
            key = field_name[len("comfy.gguf.quant."):]
            field = reader.get_field(field_name)
            custom_quant_configs[key] = json.loads(str(field.parts[field.data[-1]], "utf-8"))

    custom_quant_tensor_names = {
        tensor_name
        for key, quant_conf in custom_quant_configs.items()
        if quant_conf.get("format") == "int8_tensorwise"
        for tensor_name in (key, f"{key}_scale")
    }

    # main loading loop
    # Devices without native bf16 fall back to slow fp32 compute, so load the
    # full-precision BF16 storage tensors as fp16 there instead.
    bf16_storage_dtype = torch.bfloat16 if device_supports_bf16() else torch.float16
    state_dict = {}
    qtype_dict = {}
    for tensor_index, (sd_key, tensor) in enumerate(tensors, start=1):
        tensor_name = tensor.name
        # torch_tensor = torch.from_numpy(tensor.data) # mmap

        # NOTE: line above replaced with this block to avoid persistent numpy warning about mmap
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data) # mmap
        if dynamic_file_slice is not None:
            model_mmap, file_lock = dynamic_file_slice
            attach_dynamic_file_slice(
                torch_tensor,
                model_mmap,
                file_lock,
                tensor.data_offset,
                tensor.n_bytes,
            )

        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            # Workaround for stable-diffusion.cpp SDXL detection.
            if compat == "sd.cpp" and arch_str == "sdxl":
                if any([tensor_name.endswith(x) for x in (".proj_in.weight", ".proj_out.weight")]):
                    while len(shape) > 2 and shape[-1] == 1:
                        shape = shape[:-1]

        # add to state dict
        if dynamic and sd_key not in custom_quant_tensor_names:
            if tensor.tensor_type in {
                gguf.GGMLQuantizationType.F32,
                gguf.GGMLQuantizationType.F16,
            }:
                state_dict[sd_key] = torch_tensor.view(*shape)
                state_dict[sd_key] = apply_tensor_postprocess(
                    state_dict[sd_key], qwen35_postprocess.get(tensor_name, ()),
                )
            elif tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
                state_dict[sd_key] = torch_tensor.view(torch.bfloat16).reshape(shape).to(
                    dtype=torch.float32 if len(shape) <= 1 else bf16_storage_dtype,
                )
                state_dict[sd_key] = apply_tensor_postprocess(
                    state_dict[sd_key], qwen35_postprocess.get(tensor_name, ()),
                )
            else:
                # Text encoders generate in bf16; a different nominal dtype
                # makes ComfyUI's cast path dequantize every weight per call.
                state_dict[sd_key] = make_quantized(
                    torch_tensor, tensor.tensor_type, shape,
                    orig_dtype=bf16_storage_dtype if is_text_model else torch.float16,
                    postprocess=qwen35_postprocess.get(tensor_name, ()),
                )
        elif tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)
            state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)
            state_dict[sd_key].gguf_postprocess = qwen35_postprocess.get(tensor_name, ())
        else:
            state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)
            state_dict[sd_key].gguf_postprocess = qwen35_postprocess.get(tensor_name, ())

        if arch_str == "qwen35" and k_quant_matmul_available() and len(shape) == 2 and tensor.tensor_type in {
            gguf.GGMLQuantizationType.Q3_K,
            gguf.GGMLQuantizationType.Q4_K,
            gguf.GGMLQuantizationType.Q5_K,
            gguf.GGMLQuantizationType.Q6_K,
        }:
            state_dict[sd_key].native_matmul = True

        if isinstance(state_dict[sd_key], GGMLTensor):
            # Everything built from the reader shares the read-only file
            # mapping. Page-locking such a tensor fails (cudaHostRegister
            # rejects it) and the failed call queues an async CUDA error, so
            # the streaming paths must never try. Tensors ComfyUI copies into
            # its own host memory are fresh allocations and stay pinnable.
            state_dict[sd_key]._ggml_mmap_backed = True

        # BF16 GGUF tensors are full-precision storage, not compressed quants.
        if not dynamic and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            dtype = torch.float32 if len(shape) <= 1 else bf16_storage_dtype
            state_dict[sd_key] = dequantize_tensor(state_dict[sd_key], dtype=dtype)

        # keep track of loaded tensor types
        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1
        if progress_callback is not None:
            progress_callback(tensor_index, len(tensors))

    # print loaded tensor type counts
    logging.info("gguf qtypes: " + ", ".join(f"{k} ({v})" for k, v in qtype_dict.items()))

    # mark largest tensor for vram estimation
    qsd = {k:v for k,v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        estimate_candidates = qsd
        if arch_str == "qwen35":
            estimate_candidates = {
                k: v for k, v in qsd.items()
                if k not in {"token_embd.weight", "output.weight"}
            } or qsd
        max_key = max(estimate_candidates.keys(), key=lambda k: estimate_candidates[k].numel())
        state_dict[max_key].is_largest_weight = True

    # extra info to return
    extra = {
        "arch_str": arch_str,
        "metadata": get_gguf_metadata(reader)
    }

    # Detect custom ComfyUI native quantization metadata
    warned_unrotated_convrot = False
    for field_name in reader.fields:
        if not field_name.startswith("comfy.gguf.quant."):
            continue
        key = field_name[len("comfy.gguf.quant."):]
        field = reader.get_field(field_name)
        quant_conf = custom_quant_configs[key]
        fmt = quant_conf.get("format")

        if fmt in {"int4_compact_gemm", "int4_pytorch"}:
            raise ValueError(
                "Q4_PT GGUF files are retired because PyTorch's Ampere INT4 "
                "kernel is not performance-competitive. Reconvert as Q8_CR."
            )

        weight_key = key
        scale_key = f"{key}_scale"
        if weight_key not in state_dict or scale_key not in state_dict:
            logging.warning(f"Missing custom quant tensors for {weight_key}")
            continue

        weight_ggml = state_dict[weight_key]
        scale_ggml = state_dict[scale_key]

        if fmt == "int8_tensorwise":
            if quant_conf.get("convrot") and not quant_conf.get("weight_rotated", False):
                if not warned_unrotated_convrot:
                    logging.warning(
                        "Disabling ConvRot because this GGUF does not mark its weights "
                        "as pre-rotated. Reconvert with the current converter to enable ConvRot."
                    )
                    warned_unrotated_convrot = True
                quant_conf["convrot"] = False
                quant_conf.pop("convrot_groupsize", None)
            elif quant_conf.get("convrot"):
                groupsize = quant_conf.get("convrot_groupsize", 256)
                weight_shape = weight_ggml.shape if dynamic else weight_ggml.tensor_shape
                if weight_shape[-1] % groupsize != 0:
                    logging.warning(
                        "Disabling ConvRot for %s because %d input features are not "
                        "divisible by group size %d.",
                        weight_key,
                        weight_shape[-1],
                        groupsize,
                    )
                    quant_conf["convrot"] = False
                    quant_conf.pop("convrot_groupsize", None)

            # Convert to ComfyUI native state-dict layout
            if dynamic:
                weight = weight_ggml.view(torch.int8).reshape(weight_ggml.shape)
                scale = scale_ggml.view(torch.float32).reshape(scale_ggml.shape)
            else:
                weight = weight_ggml.data.view(torch.int8).reshape(weight_ggml.tensor_shape)
                scale = scale_ggml.data.view(torch.float32).reshape(scale_ggml.tensor_shape)

            state_dict[weight_key] = torch.nn.Parameter(weight, requires_grad=False)
            state_dict[scale_key] = torch.nn.Parameter(scale, requires_grad=False)

            layer_prefix = weight_key[:weight_key.rfind("weight")]
            quant_json = json.dumps(quant_conf)
            state_dict[f"{layer_prefix}comfy_quant"] = torch.nn.Parameter(
                torch.tensor(list(quant_json.encode("utf-8")), dtype=torch.uint8),
                requires_grad=False,
            )
            extra["gguf_quant_mode"] = "int8_convrot"

        # Retained loader adaptation for the retired Q4_PT experiment. The
        # explicit rejection above prevents this branch from being executed.
        elif fmt in {"int4_compact_gemm", "int4_pytorch"}:
            # Keep compact INT4 storage while exposing the original shape to
            # ComfyUI's model detector. It uses first.weight.shape to infer
            # Krea2's latent channel count before custom ops load the tensor.
            orig_shape = torch.Size(tuple(quant_conf["orig_shape"]))
            weight = GGMLTensor(
                weight_ggml.data.view(torch.uint8).reshape(weight_ggml.tensor_shape),
                tensor_type=weight_ggml.tensor_type,
                tensor_shape=orig_shape,
            )
            scale = scale_ggml.data.view(torch.float32).reshape(scale_ggml.tensor_shape)

            state_dict[weight_key] = torch.nn.Parameter(weight, requires_grad=False)
            state_dict[scale_key] = torch.nn.Parameter(scale, requires_grad=False)
            layer_prefix = weight_key[:weight_key.rfind("weight")]
            state_dict[f"{layer_prefix}comfy_quant"] = torch.nn.Parameter(
                torch.tensor(list(json.dumps(quant_conf).encode("utf-8")), dtype=torch.uint8),
                requires_grad=False,
            )
            extra["gguf_quant_mode"] = "int4_pytorch"

    return (state_dict, extra)

# for remapping llama.cpp -> original key names
T5_SD_MAP = {
    "enc.": "encoder.",
    ".blk.": ".block.",
    "token_embd": "shared",
    "output_norm": "final_layer_norm",
    "attn_q": "layer.0.SelfAttention.q",
    "attn_k": "layer.0.SelfAttention.k",
    "attn_v": "layer.0.SelfAttention.v",
    "attn_o": "layer.0.SelfAttention.o",
    "attn_norm": "layer.0.layer_norm",
    "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
    "ffn_up": "layer.1.DenseReluDense.wi_1",
    "ffn_down": "layer.1.DenseReluDense.wo",
    "ffn_gate": "layer.1.DenseReluDense.wi_0",
    "ffn_norm": "layer.1.layer_norm",
}

LLAMA_SD_MAP = {
    "blk.": "model.layers.",
    "attn_norm": "input_layernorm",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_v_norm.": "self_attn.v_norm.",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output.weight": "lm_head.weight",
}

QWEN35_SD_MAP = {
    "blk.": "model.language_model.layers.",
    "post_attention_norm": "post_attention_layernorm",
    "attn_norm": "input_layernorm",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_qkv": "linear_attn.in_proj_qkv",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_gate": "linear_attn.in_proj_z",
    "ssm_alpha": "linear_attn.in_proj_a",
    "ssm_beta": "linear_attn.in_proj_b",
    "ssm_conv1d": "linear_attn.conv1d",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
    "ssm_norm": "linear_attn.norm",
    "ssm_out": "linear_attn.out_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "token_embd": "model.language_model.embed_tokens",
    "output_norm": "model.language_model.norm",
    "output.weight": "lm_head.weight",
}

GEMMA3_SD_MAP = LLAMA_SD_MAP.copy()
GEMMA3_SD_MAP.update({
    "ffn_norm": "pre_feedforward_layernorm",
    "post_ffw_norm": "post_feedforward_layernorm",
    "post_attention_norm": "post_attention_layernorm",
})

CLIP_VISION_SD_MAP = {
    "mm.": "visual.merger.mlp.",
    "v.post_ln.": "visual.merger.ln_q.",
    "v.patch_embd": "visual.patch_embed.proj",
    "v.blk.": "visual.blocks.",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "attn_out.": "attn.proj.",
    "ln1.": "norm1.",
    "ln2.": "norm2.",
}

# Qwen3.5 mmproj (llama.cpp ``qwen3vl_merger`` projector): fused attn_qkv and
# a linear_fc merger. "v.blk." must be replaced before the per-block names.
CLIP_VISION_QWEN35_MAP = {
    "v.blk.": "visual.blocks.",
    "attn_qkv": "attn.qkv",
    "attn_out": "attn.proj",
    "ffn_up": "mlp.linear_fc1",
    "ffn_down": "mlp.linear_fc2",
    "ln1.": "norm1.",
    "ln2.": "norm2.",
    "mm.0.": "visual.merger.linear_fc1.",
    "mm.2.": "visual.merger.linear_fc2.",
    "v.post_ln.": "visual.merger.norm.",
    "v.patch_embd.": "visual.patch_embed.proj.",
    "v.position_embd.": "visual.pos_embed.",
}

def sd_map_replace(raw_sd, key_map):
    sd = {}
    for k,v in raw_sd.items():
        for s,d in key_map.items():
            k = k.replace(s,d)
        sd[k] = v
    return sd

def llama_permute(raw_sd, n_head, n_head_kv):
    # Reverse version of LlamaModel.permute in llama.cpp convert script
    sd = {}
    permute = lambda x,h: x.reshape(h, x.shape[0] // h // 2, 2, *x.shape[1:]).swapaxes(1, 2).reshape(x.shape)
    for k,v in raw_sd.items():
        if k.endswith(("q_proj.weight", "q_proj.bias")):
            v.data = permute(v.data, n_head)
        if k.endswith(("k_proj.weight", "k_proj.bias")):
            v.data = permute(v.data, n_head_kv)
        sd[k] = v
    return sd

def gemma3_norm_corrections(sd):
    # Reverse change from Gemma3Model modify_tensors in llama.cpp convert script
    norm_patterns = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "model.norm.weight"
    ]
    corrected = 0
    for key in list(sd.keys()):
        if any(p in key for p in norm_patterns):
            if is_quantized(sd[key]):
                sd[key] = dequantize_tensor(sd[key], dtype=torch.float32) - 1.0
            else:
                sd[key] = sd[key].float() - 1.0
            corrected += 1
    #logging.info(f"Gemma3: Applied -1 norm correction to {corrected} tensors")
    return sd


def qwen35_corrections(sd):
    """Reverse llama.cpp's Qwen3.5 conversion-time value changes."""
    for key in list(sd):
        value = sd[key]
        if key.endswith("linear_attn.A_log"):
            value = dequantize_tensor(value, dtype=torch.float32)
            sd[key] = torch.log(torch.clamp(-value, min=torch.finfo(torch.float32).tiny))
        elif key.endswith("linear_attn.dt_bias"):
            # This parameter does not pass through a GGML Linear layer, so
            # eagerly apply the inexpensive value-head layout correction.
            sd[key] = dequantize_tensor(value, dtype=torch.float32)
        elif key.endswith("linear_attn.conv1d.weight"):
            # llama.cpp squeezes PyTorch's depthwise Conv1d input-channel
            # dimension during conversion. Restore [channels, 1, kernel].
            sd[key] = dequantize_tensor(value, dtype=torch.float32).unsqueeze(1)
        elif key.endswith(("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight")):
            # The fused single-token decode kernel obtains these two gates via
            # CastBiasWeightContext rather than their GGML Linear forward.
            # Materialize only regular GGMLTensor weights; DynamicVRAM's native
            # QuantizedTensor path is already understood by ComfyUI.
            if isinstance(value, GGMLTensor) and is_quantized(value):
                sd[key] = dequantize_tensor(value, dtype=torch.float16)
        elif key.endswith("norm.weight") and ".linear_attn.norm.weight" not in key:
            # llama.cpp changes add-one RMSNorm weights to direct scale values.
            sd[key] = dequantize_tensor(value, dtype=torch.float32) - 1.0
    return sd


def fold_add_one_norms(module):
    """Bake checkpoint norms' per-call ``weight + 1`` into the weight itself.

    ComfyUI's RMSNorm adds one to its weight on every forward, which allocates a
    temporary and launches an extra kernel per norm per step (about 137 of each
    per token on a 9B encoder). The GGUF conversion already stores the plain
    scale value, so adding one once here is the same arithmetic without the
    recurring cost.

    The weight is replaced instead of updated in place: GGUF weights can be
    views of the read-only file mapping, where an in-place write faults. Packed
    (quantized) weights are skipped, since adding one to packed blocks is
    meaningless. The ``add`` flag makes this idempotent.
    """
    for child in module.modules():
        weight = getattr(child, "weight", None)
        if getattr(child, "add", None) is not True or weight is None or is_quantized(weight):
            continue
        with torch.no_grad():
            child.weight = torch.nn.Parameter(weight + 1.0, requires_grad=False)
        child.add = False


def _dynamo_disable_gguf_layers():
    """Keep every GGUF layer call opaque to dynamo.

    Packed weights advertise a compute dtype that differs from their storage
    (fp32 over packed bytes), which dynamo's fake-tensor conversion rejects, so
    the layer entry points have to be graph breaks. The elementwise math around
    them (norms, rope, gate products) is what actually fuses.
    """
    import torch._dynamo as dynamo

    for name in ("forward_comfy_cast_weights", "cast_bias_weight", "get_weight"):
        fn = getattr(GGMLLayer, name)
        if getattr(fn, "_ggml_dynamo_opaque", False):
            continue
        wrapped = dynamo.disable(fn)
        wrapped._ggml_dynamo_opaque = True
        setattr(GGMLLayer, name, wrapped)


def compile_gguf_trunk(model):
    """Compile a Qwen3.5 trunk; opt in with ``COMFYUI_GGUF_COMPILE_TRUNK=1``.

    Decode issues ~2000 kernels per token; inductor takes that to ~500 and the
    launch CPU from 58 ms to 14 ms per token, but measured wall-clock is a wash
    (6.54 vs 6.65 tok/s on the 9B encoder) because the streamed-weight,
    attention and KV work dominates, and the fused kernels round slightly
    differently, so the greedy token stream changes. It also costs 37 s
    (inductor cache warm) to 137 s (cold) per model, plus a C++ toolchain, so it
    stays off unless asked for. Returns the eager forward when the model is on
    the compiled path, ``None`` when it stays eager.
    """
    if not os.environ.get("COMFYUI_GGUF_COMPILE_TRUNK") or not hasattr(torch, "compile") or not torch.cuda.is_available():
        return None
    eager = model.forward
    try:
        _dynamo_disable_gguf_layers()
        model.forward = torch.compile(eager, dynamic=True)
    except Exception as exc:
        logging.warning(f"ComfyUI-GGUF: staying on eager execution ({exc})")
        model.forward = eager
        return None
    return eager


class MissingVisionTower(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "This Qwen3.5 text encoder was loaded without a vision tower, so it cannot take image inputs. "
            "Pick an mmproj file on the GGUF CLIP loader's mmproj input, or disconnect the image."
        )


def install_gguf_logits_support():
    """Route and cache GGUF output projections during generation.

    The llama-family text encoders (llama, qwen3, qwen3vl, qwen35, gemma4,
    ...) all generate through these shared logits/generate entry points.
    """
    import comfy.text_encoders.llama
    import comfy.text_encoders.qwen35

    base_cls = comfy.text_encoders.llama.BaseGenerate
    if getattr(base_cls, "_comfyui_gguf_logits", False):
        return

    def make_logits(original_logits):
        def logits(self, x):
            input_tensor = x[:, -1:]
            module = self.model.lm_head if hasattr(self.model, "lm_head") else self.model.embed_tokens
            # duck-typed so any GGMLLayer subclass works, even one built from a
            # separately imported copy of this package
            if getattr(module, "is_ggml_quantized", None) and module.is_ggml_quantized():
                if module.weight.supports_fused_linear(input_tensor):
                    return module.weight.fused_linear(input_tensor, module.bias)
                # Under cudaMallocAsync this dense weight would be a graph-pool
                # allocation outliving the graph that owns it, which that
                # allocator cannot account for; keep it a per-call temporary
                # there and keep the cache everywhere else.
                cache = None if (
                    torch.cuda.is_current_stream_capturing() and not triton_in_capture_usable()
                ) else getattr(self, "_gguf_logits_weight_cache", None)
                cache_key = (id(module.weight), input_tensor.device, input_tensor.dtype)
                weight = None if cache is None else cache.get(cache_key)
                if weight is None:
                    weight, _bias = module.cast_bias_weight(input_tensor)
                    if cache is not None:
                        cache[cache_key] = weight
                return torch.nn.functional.linear(input_tensor, weight, None)
            return original_logits(self, x)
        return logits

    original_generate = base_cls.generate

    def generate(self, *args, **kwargs):
        # The 9B lm_head is about 2 GiB in FP16. Keeping it only for this call
        # avoids dequantizing one billion weights for every generated token,
        # without permanently adding it to ComfyUI's managed model footprint.
        self._gguf_logits_weight_cache = {}
        try:
            return original_generate(self, *args, **kwargs)
        finally:
            self._gguf_logits_weight_cache.clear()
            del self._gguf_logits_weight_cache

    base_cls.logits = make_logits(base_cls.logits)
    base_cls.generate = generate
    base_cls._comfyui_gguf_logits = True

    qwen3_cls = getattr(comfy.text_encoders.llama, "BaseQwen3", None)
    if qwen3_cls is not None:
        qwen3_cls.logits = make_logits(qwen3_cls.logits)

    qwen35_cls = comfy.text_encoders.qwen35.Qwen35
    qwen35_cls._comfyui_gguf_logits = True
    original_init = qwen35_cls.__init__

    def init(self, config_dict, dtype, device, operations):
        original_init(self, config_dict, dtype, device, operations)
        if getattr(operations, "qwen35_text_only", False):
            # The prompt-enhancer GGUF contains no vision encoder. ComfyUI's
            # generic Qwen3.5 wrapper otherwise allocates an uninitialized
            # 1.7 GiB visual tower, crowding packed language weights off GPU.
            del self.visual
            self.visual = MissingVisionTower()

    qwen35_cls.__init__ = init

    original_qwen35_load_state_dict = qwen35_cls.load_state_dict

    def load_state_dict(self, *args, **kwargs):
        result = original_qwen35_load_state_dict(self, *args, **kwargs)
        fold_add_one_norms(self.model)
        if getattr(self, "mtp", None) is not None:
            fold_add_one_norms(self.mtp)
        return result

    qwen35_cls.load_state_dict = load_state_dict

    original_qwen35_generate = qwen35_cls.generate

    def qwen35_generate(self, *args, **kwargs):
        embeds = args[0] if args else kwargs.get("embeds")
        device = getattr(embeds, "device", None)
        eager_forward = getattr(self, "_ggml_eager_trunk_forward", None)
        if eager_forward is None and not getattr(self, "_ggml_compile_disabled", False):
            eager_forward = compile_gguf_trunk(self.model)
            self._ggml_compile_disabled = eager_forward is None
            self._ggml_eager_trunk_forward = eager_forward
        try:
            # The MTP draft graph bakes weight addresses and copies inside CUDA
            # graph capture, so its modules must already be resident and stay
            # put for the whole generate.
            if getattr(self, "mtp", None) is not None and kwargs.get("mtp", True) and device is not None:
                head = getattr(self.model, "lm_head", None) or self.model.embed_tokens
                hot = {id(m): m for m in (head, self.model.embed_tokens, *self.mtp.modules())}
                for module in hot.values():
                    module.to(device)
                # ...and the block-op constant tables have to exist before the
                # capture too; a type reached for the first time inside the
                # draft graph cannot stage its table there.
                warm_device_constants(device)
                if not triton_in_capture_usable() and any(
                    getattr(m, "is_ggml_quantized", None) and m.is_ggml_quantized() for m in hot.values()
                ):
                    # ComfyUI captures the draft head in a CUDA graph, and with
                    # the cudaMallocAsync allocator (its default on torch 2.x)
                    # a graph holding quantized GGUF weights aborts the whole
                    # process when it is replayed. Plain decoding is correct and
                    # faster in that configuration, so draw without drafts.
                    logging.warning(
                        "ComfyUI-GGUF: MTP drafting needs packed GGUF weights to be "
                        "dequantized inside ComfyUI's captured draft graph, which the "
                        "cudaMallocAsync allocator cannot replay; generating without MTP. "
                        "Start ComfyUI with --disable-cuda-malloc to enable MTP."
                    )
                    kwargs["mtp"] = False
            result = original_qwen35_generate(self, *args, **kwargs)
            self._ggml_compiled_ok = eager_forward is not None
            return result
        except Exception:
            if eager_forward is None or getattr(self, "_ggml_compiled_ok", False):
                raise
            # A compilation failure must not take the generate down.
            logging.warning("ComfyUI-GGUF: compiled trunk failed, retrying eagerly", exc_info=True)
            self.model.forward = eager_forward
            self._ggml_eager_trunk_forward = None
            self._ggml_compile_disabled = True
            return original_qwen35_generate(self, *args, **kwargs)
        finally:
            # Weights that were streamed from host memory during this generate
            # were page-locked on first use; drop those registrations while the
            # tensors are still alive rather than leaving them to outlive the
            # host buffers ComfyUI re-partitions between runs.
            for module in (self.model, getattr(self, "mtp", None)):
                if module is None:
                    continue
                for param in module.parameters():
                    unpin_host_weight(param)

    qwen35_cls.generate = qwen35_generate


install_gguf_logits_support()

def strip_quant_suffix(name):
    pattern = r"[-_.]?(?:ud-)?i?q[0-9]_[a-z0-9_\-]{1,8}$"
    match = re.search(pattern, name, re.IGNORECASE)
    if match:
        name = name[:match.start()]
    return name

def find_mmproj(path):
    """The mmproj GGUF next to ``path`` whose name contains the text encoder's name, if any."""
    # get name to match w/o quant suffix
    tenc_fname = os.path.basename(path)
    tenc = strip_quant_suffix(os.path.splitext(tenc_fname)[0].lower())

    target = []
    root = os.path.dirname(path)
    for fname in os.listdir(root):
        name, ext = os.path.splitext(fname)
        if ext.lower() == ".gguf" and "mmproj" in name.lower() and tenc in name.lower():
            target.append(fname)

    if len(target) > 1:
        logging.warning(f"Ambiguous mmproj for text encoder '{tenc_fname}', will use first match.")
    return os.path.join(root, target[0]) if target else None

def gguf_mmproj_loader(path, arch, dynamic=False):
    # Reverse version of Qwen2VLVisionModel.modify_tensors
    logging.info(f"Using mmproj '{os.path.basename(path)}'.")
    vsd, _ = gguf_sd_loader(path, is_text_model=True, dynamic=dynamic)

    # concat 4D to 5D
    if "v.patch_embd.weight.1" in vsd:
        w1 = dequantize_tensor(vsd.pop("v.patch_embd.weight"), dtype=torch.float32)
        w2 = dequantize_tensor(vsd.pop("v.patch_embd.weight.1"), dtype=torch.float32)
        vsd["v.patch_embd.weight"] = torch.stack([w1, w2], dim=2)

    if arch == "qwen35":
        return sd_map_replace(vsd, CLIP_VISION_QWEN35_MAP)

    # run main replacement
    vsd = sd_map_replace(vsd, CLIP_VISION_SD_MAP)

    # handle split Q/K/V
    if "visual.blocks.0.attn_q.weight" in vsd:
        attns = {}
        # filter out attentions + group
        for k,v in vsd.items():
            if any(x in k for x in ["attn_q", "attn_k", "attn_v"]):
                k_attn, k_name = k.rsplit(".attn_", 1)
                k_attn += ".attn.qkv." + k_name.split(".")[-1]
                if k_attn not in attns:
                    attns[k_attn] = {}
                attns[k_attn][k_name] = dequantize_tensor(
                    v, dtype=(torch.bfloat16 if is_quantized(v) else torch.float16)
                )

        # recombine
        for k,v in attns.items():
            suffix = k.split(".")[-1]
            vsd[k] = torch.cat([
                v[f"q.{suffix}"],
                v[f"k.{suffix}"],
                v[f"v.{suffix}"],
            ], dim=0)
        del attns

    return vsd

def gguf_tokenizer_loader(path, temb_shape):
    # convert gguf tokenizer to spiece
    logging.info("Attempting to recreate sentencepiece tokenizer from GGUF file metadata...")
    try:
        from sentencepiece import sentencepiece_model_pb2 as model
    except ImportError:
        raise ImportError("Please make sure sentencepiece and protobuf are installed.\npip install sentencepiece protobuf")
    spm = model.ModelProto()

    reader = gguf.GGUFReader(path)

    if get_field(reader, "tokenizer.ggml.model", str) == "t5":
        if temb_shape == (256384, 4096): # probably UMT5
            spm.trainer_spec.model_type == 1 # Unigram (do we have a T5 w/ BPE?)
        else:
            raise NotImplementedError("Unknown model, can't set tokenizer!")
    else:
        raise NotImplementedError("Unknown model, can't set tokenizer!")

    spm.normalizer_spec.add_dummy_prefix = get_field(reader, "tokenizer.ggml.add_space_prefix", bool)
    spm.normalizer_spec.remove_extra_whitespaces = get_field(reader, "tokenizer.ggml.remove_extra_whitespaces", bool)

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    scores = get_list_field(reader, "tokenizer.ggml.scores", float)
    toktypes = get_list_field(reader, "tokenizer.ggml.token_type", int)

    for idx, (token, score, toktype) in enumerate(zip(tokens, scores, toktypes)):
        # # These aren't present in the original?
        # if toktype == 5 and idx >= temb_shape[0]%1000):
        #     continue

        piece = spm.SentencePiece()
        piece.piece = token
        piece.score = score
        piece.type = toktype
        spm.pieces.append(piece)

    # unsure if any of these are correct
    spm.trainer_spec.byte_fallback = True
    spm.trainer_spec.vocab_size = len(tokens) # split off unused?
    spm.trainer_spec.max_sentence_length = 4096
    spm.trainer_spec.eos_id = get_field(reader, "tokenizer.ggml.eos_token_id", int)
    spm.trainer_spec.pad_id = get_field(reader, "tokenizer.ggml.padding_token_id", int)

    logging.info(f"Created tokenizer with vocab size of {len(spm.pieces)}")
    del reader
    return torch.ByteTensor(list(spm.SerializeToString()))

def gguf_tekken_tokenizer_loader(path, temb_shape):
    # convert ggml (hf) tokenizer metadata to tekken/comfy data
    logging.info("Attempting to recreate tekken tokenizer from GGUF file metadata...")
    import json
    import base64
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    reader = gguf.GGUFReader(path)

    model_str = get_field(reader, "tokenizer.ggml.model", str)
    if model_str == "gpt2":
        if temb_shape == (131072, 5120): # probably Mistral
            data = {
                "config": {"num_vocab_tokens": 150000, "default_vocab_size": 131072},
                "vocab": [],
                "special_tokens": [],
            }
        else:
            raise NotImplementedError("Unknown model, can't set tokenizer!")
    else:
        raise NotImplementedError("Unknown model, can't set tokenizer!")

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    toktypes = get_list_field(reader, "tokenizer.ggml.token_type", int)

    decoder = {v: k for k, v in bytes_to_unicode().items()}
    for idx, (token, toktype) in enumerate(zip(tokens, toktypes)):
        if toktype == 3:
            data["special_tokens"].append(
                {'rank': idx, 'token_str': token, 'is_control': True}
            )
        else:
            tok = bytes([decoder[char] for char in token])
            data["vocab"].append({
                "rank": len(data["vocab"]),
                "token_bytes": base64.b64encode(tok).decode("ascii"),
                "token_str": tok.decode("utf-8", errors="replace") # ?
            })

    logging.info(f"Created tekken tokenizer with vocab size of {len(data['vocab'])} (+{len(data['special_tokens'])})")
    del reader
    return torch.ByteTensor(list(json.dumps(data).encode('utf-8')))

def gguf_gemma3_tokenizer_loader(path):
    #TODO: merge into gguf_tokenizer_loader
    logging.info("Attempting to recreate sentencepiece tokenizer from GGUF file metadata...")
    try:
        from sentencepiece import sentencepiece_model_pb2 as model
    except ImportError:
        raise ImportError("Please install sentencepiece and protobuf.\npip install sentencepiece protobuf")
    spm = model.ModelProto()
    reader = gguf.GGUFReader(path)

    spm.normalizer_spec.name = "identity"
    spm.normalizer_spec.add_dummy_prefix = False
    spm.trainer_spec.model_type = 2
    spm.trainer_spec.input_format = "tsv"
    spm.trainer_spec.byte_fallback = True
    spm.trainer_spec.max_sentence_length = 4192
    spm.trainer_spec.bos_piece = "<bos>"

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    scores = get_list_field(reader, "tokenizer.ggml.scores", float)
    toktype = get_list_field(reader, "tokenizer.ggml.token_type", int)

    if not tokens or not scores or not toktype:
        raise ValueError("Missing tokenizer metadata")

    for idx in range(len(tokens)):
        piece = spm.SentencePiece()
        piece.piece = tokens[idx]
        if idx == 3:  # UNK position
            piece.type = 2  # UNK Token
            piece.score = 0.0 # UNK Score
        else:
            piece.type = toktype[idx]
            piece.score = scores[idx]
        spm.pieces.append(piece)

    spm.trainer_spec.vocab_size = len(spm.pieces)
    logging.info(f"Created tokenizer with vocab size of {len(spm.pieces)}")

    del reader
    return torch.ByteTensor(list(spm.SerializeToString()))

def inject_qwen3vl_detection_markers(sd):
    """Add visual sentinels when a llama.cpp Qwen3-VL GGUF excludes its vision tower."""
    ln_key = "model.layers.0.input_layernorm.weight"
    lm_hidden = int(sd[ln_key].shape[0]) if ln_key in sd else 2560
    vis_hidden = 1024 if lm_hidden == 2560 else 1152
    merge_dim = vis_hidden * 4  # spatial_merge_size=2

    if lm_hidden == 5120:
        # MiniMax H3 uses the truncated Qwen3-VL-32B encoder. Its detector
        # deliberately checks this unprefixed visual key plus layer 49.
        marker_key = "visual.deepstack_merger_list.0.norm.weight"
    else:
        marker_key = "model.visual.deepstack_merger_list.0.norm.weight"

    sd[marker_key] = torch.zeros(merge_dim)
    if lm_hidden != 5120:
        sd["model.visual.merger.linear_fc2.weight"] = torch.zeros(lm_hidden, merge_dim)
    logging.info(
        "qwen3vl GGUF: injected visual marker tensor "
        "(lm_hidden=%d, merge_dim=%d)",
        lm_hidden,
        merge_dim,
    )

MTP_ROOT_ALIASES = {
    # llama.cpp "nextn" speculative-head names -> HF-canonical MTP names
    "nextn.eh_proj.weight": "fc.weight",
    "nextn.enorm.weight": "pre_fc_norm_embedding.weight",
    "nextn.hnorm.weight": "pre_fc_norm_hidden.weight",
    "nextn.shared_head_norm.weight": "norm.weight",
    "fc.weight": "fc.weight",
    "norm.weight": "norm.weight",
    "pre_fc_norm_embedding.weight": "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight": "pre_fc_norm_hidden.weight",
}

MTP_LAYER_ALIASES = {
    # llama.cpp block names -> HF-canonical MTP layer names
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}


def normalize_mtp_keys(sd):
    """Map any known MTP tensor naming onto HF-canonical ``mtp.*`` names.

    Understands HF names (``mtp.layers.0.self_attn.q_proj.weight``), llama.cpp
    block names (``blk.32.nextn.eh_proj.weight``, ``blk.32.attn_q.weight``)
    and the prefixed variants used by MTP-only sidecar files.
    """
    out = {}
    for key, value in sd.items():
        name = key
        for prefix in ("model.language_model.mtp.", "language_model.mtp.",
                       "transformer.mtp.", "model.mtp."):
            if name.startswith(prefix):
                name = "mtp." + name[len(prefix):]
                break
        if name.startswith("blk."):
            name = name.split(".", 2)[2]
        if name.startswith("mtp."):
            name = name[len("mtp."):]
        root = MTP_ROOT_ALIASES.get(name)
        if root is not None:
            out["mtp." + root] = value
            continue
        index = "0"
        if name.startswith("layers."):
            name = name[len("layers."):]
            index, _, name = name.partition(".")
        out[f"mtp.layers.{index}.{MTP_LAYER_ALIASES.get(name, name)}"] = value
    return out


def split_mtp_tensors(sd):
    """Split an embedded MTP draft head off a full llama.cpp qwen35 state dict.

    llama.cpp stores the MTP head as an extra block (``blk.N``) whose
    ``nextn.*`` keys hold the draft projections. Returns ``(mtp_sd, rest_sd)``
    with the MTP tensors renamed onto HF-canonical ``mtp.*`` names.
    """
    mtp_block = None
    for key in sd:
        if ".nextn." in key:
            mtp_block = key.split(".")[1]
            break
    if mtp_block is None and not any(k.startswith("mtp") or ".mtp." in k for k in sd):
        return {}, sd
    prefix = f"blk.{mtp_block}." if mtp_block is not None else None
    mtp_sd = {}
    rest_sd = {}
    for key, value in sd.items():
        if (prefix is not None and key.startswith(prefix)) or key.startswith("mtp") or ".mtp." in key:
            mtp_sd.update(normalize_mtp_keys({key: value}))
        else:
            rest_sd[key] = value
    return mtp_sd, rest_sd


def gguf_mtp_loader(path):
    """Load MTP draft-head weights from a GGUF file of any naming convention.

    Returns an HF-canonical ``mtp.*`` state dict ready to merge into a text
    encoder state dict (which enables ComfyUI's MTP generation path).
    """
    reader = gguf.GGUFReader(path)
    sd = {}
    for tensor in reader.tensors:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data)
        shape = get_orig_shape(reader, tensor.name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
        if tensor.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            sd[tensor.name] = torch_tensor.view(*shape)
        elif tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            sd[tensor.name] = torch_tensor.view(torch.bfloat16).reshape(shape)
        else:
            sd[tensor.name] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)
    mtp_sd, rest_sd = split_mtp_tensors(sd)
    if not mtp_sd:
        # sidecar files hold only MTP tensors; accept them wholesale
        mtp_sd = normalize_mtp_keys(sd)
    if "mtp.fc.weight" not in mtp_sd:
        raise ValueError(f"No MTP weights found in GGUF file: {path}")
    # Note: the draft head's norms are kept exactly as stored in the file. The
    # encoder itself stores add-one norms as scale values (see
    # qwen35_corrections), and applying that same shift here was tried and
    # measured worse on the graft: 1.62 vs 1.85 accepted tokens per verify step
    # for the same head on a prompt-enhancer trunk, so the file values win.
    logging.info(f"loaded {len(mtp_sd)} MTP tensors from {os.path.basename(path)}")
    return mtp_sd


def gguf_clip_loader(path, dynamic=False, progress_callback=None, mmproj="auto"):
    """``mmproj`` is "auto" (a name-matched file next to ``path``), "none", or a file path."""
    sd, extra = gguf_sd_loader(
        path,
        is_text_model=True,
        dynamic=dynamic,
        progress_callback=progress_callback,
    )
    arch = extra.get("arch_str", None)
    if arch in {"t5", "t5encoder"}:
        temb_key = "token_embd.weight"
        if temb_key in sd and sd[temb_key].shape == (256384, 4096):
            # non-standard Comfy-Org tokenizer
            sd["spiece_model"] = gguf_tokenizer_loader(path, sd[temb_key].shape)
            # TODO: dequantizing token embed here is janky but otherwise we OOM due to tensor being massive.
            logging.warning(f"Dequantizing {temb_key} to prevent runtime OOM.")
            sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)
        sd = sd_map_replace(sd, T5_SD_MAP)
    elif arch in {"llama", "qwen2vl", "qwen3", "qwen3vl", "qwen35", "gemma3"}:
        # TODO: pass model_options["vocab_size"] to loader somehow
        temb_key = "token_embd.weight"
        if temb_key in sd and sd[temb_key].shape[0] >= (64 * 1024):
            if arch == "llama" and sd[temb_key].shape == (131072, 5120):
                # non-standard Comfy-Org tokenizer
                sd["tekken_model"] = gguf_tekken_tokenizer_loader(path, sd[temb_key].shape)
            elif arch == "gemma3":
                sd["spiece_model"] = gguf_gemma3_tokenizer_loader(path)
            # Qwen3.5 K-quant embeddings gather and dequantize only the rows
            # requested by the tokenizer; other architectures retain the
            # compatibility fallback that materializes the whole table.
            if arch != "qwen35":
                logging.warning(f"Dequantizing {temb_key} to prevent runtime OOM.")
                sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)
        if arch == "gemma3":
            sd = sd_map_replace(sd, GEMMA3_SD_MAP)
            sd = gemma3_norm_corrections(sd)
        elif arch == "qwen35":
            mtp_sd, sd = split_mtp_tensors(sd)
            sd = sd_map_replace(sd, QWEN35_SD_MAP)
            sd.update(mtp_sd)
            sd = qwen35_corrections(sd)
            mmproj_path = find_mmproj(path) if mmproj == "auto" else None if mmproj == "none" else mmproj
            if mmproj_path is not None:
                sd.update(gguf_mmproj_loader(mmproj_path, arch, dynamic=dynamic))
        else:
            sd = sd_map_replace(sd, LLAMA_SD_MAP)
        if arch == "llama":
            sd = llama_permute(sd, 32, 8) # L3 / Mistral
        if arch == "qwen2vl" and mmproj != "none":
            mmproj_path = find_mmproj(path) if mmproj == "auto" else mmproj
            if mmproj_path is None:
                logging.error(f"Error: Can't find mmproj file for '{os.path.basename(path)}'! Qwen-Image-Edit will be broken!")
            else:
                sd.update(gguf_mmproj_loader(mmproj_path, arch, dynamic=dynamic))
        if arch == "qwen3vl" and "model.visual.deepstack_merger_list.0.norm.weight" not in sd:
            # Standard llama.cpp Qwen3-VL GGUFs omit the visual tower. Without it,
            # detect_te_model() mis-classifies the state dict as a Qwen3 LM instead
            # of Qwen3-VL. MiniMax H3 additionally uses Qwen3-VL-32B truncated to
            # 50 layers, whose detector relies on an unprefixed visual marker.
            # Inject zero sentinel tensors with shapes that exactly match the model
            # parameters so that load_state_dict(strict=False) doesn't raise a size
            # mismatch error while still satisfying detect_te_model()'s key checks.
            inject_qwen3vl_detection_markers(sd)
    elif arch == "ideogram":
        # Dequantize Ideogram model for inference
        logging.info("Dequantizing Ideogram model for inference...")
        # Use BF16 to save VRAM while maintaining quality, but fall back to FP16
        # on devices that don't support bf16 (avoids slow fp32 compute fallback).
        target_dtype = torch.bfloat16 if device_supports_bf16() else torch.float16
        dequantized_count = 0
        for key in list(sd.keys()):
            if is_quantized(sd[key]):
                sd[key] = dequantize_tensor(sd[key], dtype=target_dtype)
                dequantized_count += 1
        logging.info(f"Dequantized {dequantized_count} tensors for Ideogram model ({target_dtype})")
    else:
        pass
    return sd
