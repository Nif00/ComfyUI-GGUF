# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import os
import gguf
import torch
from tqdm import tqdm

try:
    import triton
    import triton.language as tl
except ImportError:  # Optional; the PyTorch block ops below remain the fallback.
    triton = None
    tl = None


TORCH_COMPATIBLE_QTYPES = (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)

# 16-entry IQ lookup table, shared by the block ops and the Triton kernels.
_KVALUES_VALUES = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

_device_constants = {}
_triton_capture_ok = None


def device_constant(values, device, dtype):
    """Small constant table on ``device``, safe to create inside graph capture.

    ``torch.tensor(values, device=...)`` stages through pageable host memory and
    CUDA graph capture rejects that copy outright, so a constant that first
    shows up mid-capture is staged through a pinned buffer instead. Tables are
    cached per device, so this only matters for the types a run reaches for the
    first time while capturing.
    """
    key = (tuple(values), str(device), dtype)
    table = _device_constants.get(key)
    if table is None:
        host = torch.tensor(values, dtype=dtype)
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            # Capture rejects both pageable copies and synchronous ones, so the
            # staging buffer is pinned and the transfer is asynchronous.
            table = host.pin_memory().to(device, non_blocking=True)
        else:
            table = host.to(device)
        _device_constants[key] = table
    return table


# Every shift pattern the block ops paste across a block.
_SHIFT_VALUES = (
    [0, 4],
    [0, 2, 4, 6],
    list(range(8)),
    [2 * i for i in range(8)],
)


def warm_device_constants(device):
    """Build the block-op constant tables before any graph capture starts.

    Types first reached while a draft graph is being captured would otherwise
    have to stage their tables mid-capture; doing it up front keeps the capture
    itself free of host traffic.
    """
    if device is None or getattr(device, "type", None) != "cuda":
        return
    for values in _SHIFT_VALUES:
        device_constant(values, device, torch.uint8)
    get_kvalues(device)


def apply_tensor_postprocess(tensor, transforms=()):
    """Apply reversible layout changes recorded on a GGUF tensor."""
    for transform in transforms:
        kind, dim, start, num_k_heads, num_v_per_k, head_dim = transform
        if kind != "qwen35_inverse_v_heads":
            raise ValueError(f"Unknown GGUF tensor postprocess operation: {kind!r}")

        # llama.cpp stores Qwen3.5 value heads in tiled order so GGML can use a
        # broadcast. PyTorch's reference implementation expects them grouped
        # by K head, so undo [R,K,D] -> [K,R,D] after dequantization.
        size = tensor.shape[dim]
        value = tensor.narrow(dim, start, size - start)
        shape = list(value.shape)
        if shape[dim] != num_k_heads * num_v_per_k * head_dim:
            raise ValueError(
                "Invalid Qwen3.5 value-head dimension: "
                f"got {shape[dim]}, expected {num_k_heads * num_v_per_k * head_dim}"
            )
        tiled_shape = shape[:dim] + [num_v_per_k, num_k_heads, head_dim] + shape[dim + 1:]
        value = value.reshape(tiled_shape)
        perm = list(range(len(tiled_shape)))
        perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
        value = value.permute(perm).contiguous().reshape(shape)
        if start:
            tensor = torch.cat((tensor.narrow(dim, 0, start), value), dim=dim)
        else:
            tensor = value
    return tensor

def is_torch_compatible(tensor):
    return tensor is None or getattr(tensor, "tensor_type", None) in TORCH_COMPATIBLE_QTYPES

def is_quantized(tensor):
    return not is_torch_compatible(tensor)

def dequantize_tensor(tensor, dtype=None, dequant_dtype=None):
    qtype = getattr(tensor, "tensor_type", None)
    oshape = getattr(tensor, "tensor_shape", tensor.shape)
    postprocess = getattr(tensor, "gguf_postprocess", ())

    if qtype in TORCH_COMPATIBLE_QTYPES:
        result = tensor.to(dtype)
    elif qtype == gguf.GGMLQuantizationType.BF16:
        result = torch.Tensor(tensor.data.view(torch.bfloat16).reshape(oshape))
        result = result if dtype is None or dtype == torch.bfloat16 else result.to(dtype)
    elif qtype in dequantize_functions:
        dequant_dtype = dtype if dequant_dtype == "target" else dequant_dtype
        result = dequantize(tensor.data, qtype, oshape, dtype=dequant_dtype).to(dtype)
    else:
        # this is incredibly slow
        tqdm.write(f"Falling back to numpy dequant for qtype: {getattr(qtype, 'name', repr(qtype))}")
        new = gguf.quants.dequantize(tensor.cpu().numpy(), qtype)
        result = torch.from_numpy(new).to(tensor.device, dtype=dtype)
    # Tensor subclass propagation can leave fully dequantized values branded as
    # GGMLTensor, causing later code to interpret their float storage as packed
    # quant bytes a second time. Materialization must return a plain Tensor.
    if type(result) is not torch.Tensor and isinstance(result, torch.Tensor):
        result = result.as_subclass(torch.Tensor)
    return apply_tensor_postprocess(result, postprocess)

def dequantize(data, qtype, oshape, dtype=None):
    """
    Dequantize tensor back to usable shape/dtype
    """
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    dequantize_blocks = dequantize_functions[qtype]

    rows = data.reshape(
        (-1, data.shape[-1])
    ).view(torch.uint8)

    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape)

def to_uint32(x):
    # no uint32 :(
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)

def to_uint16(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8).unsqueeze(1)

def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)

# Full weights #
def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)

# Legacy Quants #
def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return (d * x)

def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> device_constant([0, 4], d.device, torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))

    qs = (ql | (qh << 4))
    return (d * qs) + m

def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qh, qs = split_block_dims(blocks, 2, 4)
    d  = d.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> device_constant([0, 4], d.device, torch.uint8).reshape(1, 1, 2, 1)

    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)

    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return (d * qs)

def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qs = split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> device_constant([0, 4], d.device, torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)

    return (d * qs) + m

def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d  = d.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> device_constant([0, 4], d.device, torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return (d * qs)

# K Quants #
QK_K = 256
K_SCALE_SIZE = 12

def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))

    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)

    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))

def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]


    ql, qh, scales, d, = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)

    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

    ql = ql.reshape((n_blocks, -1, 1, 64)) >> device_constant([0, 4], d.device, torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> device_constant([0, 2, 4, 6], d.device, torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))

    return (d * q).reshape((n_blocks, QK_K))

def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)

    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> device_constant([0, 4], d.device, torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> device_constant([i for i in range(8)], d.device, torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4))

    return (d * q - dm).reshape((n_blocks, QK_K))

def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> device_constant([0, 4], d.device, torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))

    return (d * qs - dm).reshape((n_blocks, QK_K))

def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)

    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> device_constant([0, 4], d.device, torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> device_constant([0, 2, 4, 6], d.device, torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = (scales.to(torch.int8) - 32)

    dl = (d * scales).reshape((n_blocks, 16, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> device_constant([0, 2, 4, 6], d.device, torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> device_constant([i for i in range(8)], d.device, torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8))

    return (dl * q).reshape((n_blocks, QK_K))

def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    # (n_blocks, 16, 1)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))

    shift = device_constant([0, 2, 4, 6], d.device, torch.uint8).reshape((1, 1, 4, 1))

    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml

    return qs.reshape((n_blocks, -1))

# IQ quants

def dequantize_blocks_IQ4_NL(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size//2)) >> device_constant([0, 4], d.device, torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 1)).to(torch.int64)

    kvalues = get_kvalues(qs.device).expand(*qs.shape[:-1], 16)
    qs = torch.gather(kvalues, dim=-1, index=qs).reshape((n_blocks, -1))
    del kvalues # should still be view, but just to be safe

    return (d * qs)

def dequantize_blocks_IQ4_XS(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, scales_h, scales_l, qs = split_block_dims(blocks, 2, 2, QK_K // 64)
    d = d.view(torch.float16).to(dtype)
    scales_h = to_uint16(scales_h)

    shift_a = device_constant([0, 4], d.device, torch.uint8).reshape((1, 1, 2))
    shift_b = device_constant([2 * i for i in range(QK_K // 32)], d.device, torch.uint8).reshape((1, -1, 1))

    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> shift_a.reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, -1, 1)) >> shift_b.reshape((1, -1, 1))

    scales_l = scales_l.reshape((n_blocks, -1)) & 0x0F
    scales_h = scales_h.reshape((n_blocks, -1)).to(torch.uint8) & 0x03

    scales = (scales_l | (scales_h << 4)).to(torch.int8) - 32
    dl = (d * scales.to(dtype)).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 16)) >> shift_a.reshape((1, 1, 2, 1))
    qs = qs.reshape((n_blocks, -1, 32, 1)) & 0x0F

    kvalues = get_kvalues(qs.device).expand(*qs.shape[:-1], 16)
    qs = torch.gather(kvalues, dim=-1, index=qs.to(torch.int64)).reshape((n_blocks, -1, 32))
    del kvalues # see IQ4_NL
    del shift_a
    del shift_b

    return (dl * qs).reshape((n_blocks, -1))

# Fused CUDA kernels. One ggml block decoder serves both the dequantization
# kernel below and the small-batch matmul kernel in quant_matmul.py. Format ids
# keep the device-code switch small and stable.
_Q2_K, _Q3_K, _Q4_K, _Q5_K, _Q6_K = 0, 1, 2, 3, 4
_Q4_0, _Q4_1, _Q5_0, _Q5_1, _Q8_0 = 5, 6, 7, 8, 9
_IQ4_NL, _IQ4_XS = 10, 11

CUDA_QTYPES = {
    gguf.GGMLQuantizationType.Q2_K: _Q2_K,
    gguf.GGMLQuantizationType.Q3_K: _Q3_K,
    gguf.GGMLQuantizationType.Q4_K: _Q4_K,
    gguf.GGMLQuantizationType.Q5_K: _Q5_K,
    gguf.GGMLQuantizationType.Q6_K: _Q6_K,
    gguf.GGMLQuantizationType.Q4_0: _Q4_0,
    gguf.GGMLQuantizationType.Q4_1: _Q4_1,
    gguf.GGMLQuantizationType.Q5_0: _Q5_0,
    gguf.GGMLQuantizationType.Q5_1: _Q5_1,
    gguf.GGMLQuantizationType.Q8_0: _Q8_0,
    gguf.GGMLQuantizationType.IQ4_NL: _IQ4_NL,
    gguf.GGMLQuantizationType.IQ4_XS: _IQ4_XS,
}

_TL_DTYPES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}

# Formats whose 256-value blocks decode with contiguous field loads.
WIDE_FORMATS = frozenset({_Q2_K, _Q3_K, _Q4_K, _Q5_K, _Q6_K, _IQ4_XS})

_DEQUANT_NUM_WARPS = 8


def get_kvalues(device):
    """Per-device copy of the IQ4 lookup table consumed by the kernels."""
    return device_constant(_KVALUES_VALUES, device, torch.int8)

if triton is not None:
    @triton.jit
    def load_f16(q_ptr, offset):
        bits = tl.load(q_ptr + offset).to(tl.uint16)
        bits |= tl.load(q_ptr + offset + 1).to(tl.uint16) << 8
        return bits.to(tl.float16, bitcast=True).to(tl.float32)


    @triton.jit
    def load_u16(q_ptr, offset):
        value = tl.load(q_ptr + offset).to(tl.uint16)
        value |= tl.load(q_ptr + offset + 1).to(tl.uint16) << 8
        return value


    @triton.jit
    def load_u32(q_ptr, offset):
        value = tl.load(q_ptr + offset).to(tl.uint32)
        value |= tl.load(q_ptr + offset + 1).to(tl.uint32) << 8
        value |= tl.load(q_ptr + offset + 2).to(tl.uint32) << 16
        value |= tl.load(q_ptr + offset + 3).to(tl.uint32) << 24
        return value


    @triton.jit
    def decode_block_value(q_ptr, block_base, within, kvalues_ptr, TYPE: tl.constexpr):
        """Decode one packed ggml element; `within` is its value index in the block.

        The index-to-byte mappings match ggml's reference dequantize_row_*
        functions and therefore the PyTorch block ops above.
        """
        if TYPE == 0:  # Q2_K
            scale = tl.load(q_ptr + block_base + (within // 16))
            q_byte = tl.load(q_ptr + block_base + 16 + (within // 128) * 32 + (within % 32))
            quant = (q_byte >> (((within // 32) % 4) * 2)) & 3
            d = load_f16(q_ptr, block_base + 80)
            dmin = load_f16(q_ptr, block_base + 82)
            return (d * (scale & 15).to(tl.float32) * quant.to(tl.float32)
                    - dmin * (scale >> 4).to(tl.float32))

        elif TYPE == 1:  # Q3_K
            q_byte = tl.load(q_ptr + block_base + 32 + (within // 128) * 32 + (within % 32))
            q_low = (q_byte >> (((within % 128) // 32) * 2)) & 3
            h_byte = tl.load(q_ptr + block_base + (within % 32))
            q_high = (((h_byte >> (within // 32)) & 1) ^ 1).to(tl.int32)
            quant = q_low.to(tl.int32) - (q_high << 2)

            scale_group = within // 16
            low_byte = tl.load(q_ptr + block_base + 96 + (scale_group % 8))
            low = (low_byte >> tl.where(scale_group < 8, 0, 4)) & 15
            high_byte = tl.load(q_ptr + block_base + 104 + (scale_group % 4))
            high = (high_byte >> (2 * (scale_group // 4))) & 3
            scale = (low | (high << 4)).to(tl.int32) - 32
            d = load_f16(q_ptr, block_base + 108)
            return d * scale.to(tl.float32) * quant.to(tl.float32)

        elif TYPE == 2 or TYPE == 3:  # Q4_K / Q5_K
            scale_group = within // 32
            q_array = scale_group // 2
            q_shift = (scale_group % 2) * 4
            q_pos = within % 32
            data_offset: tl.constexpr = 16 if TYPE == 2 else 48
            q_byte = tl.load(q_ptr + block_base + data_offset + q_array * 32 + q_pos)
            quant = (q_byte >> q_shift) & 15
            if TYPE == 3:
                h_byte = tl.load(q_ptr + block_base + 16 + q_pos)
                quant |= ((h_byte >> scale_group) & 1) << 4

            first = tl.load(q_ptr + block_base + 4 + (scale_group % 4))
            middle = tl.load(q_ptr + block_base + 8 + (scale_group % 4))
            shared = tl.load(q_ptr + block_base + 12 + (scale_group % 4))
            scale = tl.where(
                scale_group < 4,
                first & 63,
                (shared & 15) | ((first >> 2) & 48),
            )
            minimum = tl.where(
                scale_group < 4,
                middle & 63,
                (shared >> 4) | ((middle >> 2) & 48),
            )
            d = load_f16(q_ptr, block_base)
            dmin = load_f16(q_ptr, block_base + 2)
            return (d * scale.to(tl.float32) * quant.to(tl.float32)
                    - dmin * minimum.to(tl.float32))

        elif TYPE == 4:  # Q6_K
            low_group = within // 128
            low_shift = ((within % 128) // 64) * 4
            low_byte = tl.load(q_ptr + block_base + low_group * 64 + (within % 64))
            q_low = (low_byte >> low_shift) & 15
            high_byte = tl.load(q_ptr + block_base + 128 + low_group * 32 + (within % 32))
            q_high = (high_byte >> (((within % 128) // 32) * 2)) & 3
            quant = (q_low | (q_high << 4)).to(tl.int32) - 32
            scale = tl.load(q_ptr + block_base + 192 + (within // 16)).to(tl.int8).to(tl.int32)
            d = load_f16(q_ptr, block_base + 208)
            return d * scale.to(tl.float32) * quant.to(tl.float32)

        elif TYPE == 5 or TYPE == 10:  # Q4_0 / IQ4_NL
            d = load_f16(q_ptr, block_base)
            nibble_byte = tl.load(q_ptr + block_base + 2 + (within % 16))
            index = ((nibble_byte >> ((within // 16) * 4)) & 15).to(tl.int32)
            if TYPE == 5:
                return d * (index.to(tl.float32) - 8.0)
            return d * tl.load(kvalues_ptr + index).to(tl.float32)

        elif TYPE == 6:  # Q4_1
            d = load_f16(q_ptr, block_base)
            m = load_f16(q_ptr, block_base + 2)
            nibble_byte = tl.load(q_ptr + block_base + 4 + (within % 16))
            quant = (nibble_byte >> ((within // 16) * 4)) & 15
            return d * quant.to(tl.float32) + m

        elif TYPE == 7 or TYPE == 8:  # Q5_0 / Q5_1
            d = load_f16(q_ptr, block_base)
            has_min: tl.constexpr = TYPE == 8
            qh = load_u32(q_ptr, block_base + (4 if has_min else 2))
            nibble_byte = tl.load(q_ptr + block_base + (8 if has_min else 6) + (within % 16))
            quant = ((nibble_byte >> ((within // 16) * 4)) & 15).to(tl.int32)
            quant |= (((qh >> within) & 1) << 4).to(tl.int32)
            if has_min:
                m = load_f16(q_ptr, block_base + 2)
                return d * quant.to(tl.float32) + m
            return d * (quant.to(tl.float32) - 16.0)

        elif TYPE == 9:  # Q8_0
            d = load_f16(q_ptr, block_base)
            quant = tl.load(q_ptr + block_base + 2 + within).to(tl.int8)
            return d * quant.to(tl.float32)

        else:  # IQ4_XS
            scale_group = within // 32
            d = load_f16(q_ptr, block_base)
            scales_h = load_u16(q_ptr, block_base + 2)
            low_byte = tl.load(q_ptr + block_base + 4 + (scale_group // 2))
            low = (low_byte >> ((scale_group % 2) * 4)) & 15
            high = (scales_h >> (scale_group * 2)) & 3
            scale = (low | (high << 4)).to(tl.int32) - 32
            nibble_byte = tl.load(q_ptr + block_base + 8 + scale_group * 16 + (within % 16))
            index = ((nibble_byte >> (((within % 32) // 16) * 4)) & 15).to(tl.int32)
            return d * scale.to(tl.float32) * tl.load(kvalues_ptr + index).to(tl.float32)


    @triton.jit
    def expand_group(v, ROWS: tl.constexpr, COLS: tl.constexpr):
        """Expand a [ROWS] per-group value to [ROWS * COLS], repeating per col."""
        return tl.reshape(
            tl.broadcast_to(tl.reshape(v, (ROWS, 1)), (ROWS, COLS)), (ROWS * COLS,),
        )


    @triton.jit
    def expand_along(v, ROWS: tl.constexpr, COLS: tl.constexpr):
        """Broadcast a [COLS] block field to [ROWS * COLS], one row per group."""
        return tl.reshape(
            tl.broadcast_to(tl.reshape(v, (1, COLS)), (ROWS, COLS)), (ROWS * COLS,),
        )


    @triton.jit
    def expand_pairs(v, ROWS: tl.constexpr, COLS: tl.constexpr):
        """Expand [ROWS * COLS] bytes into nibble pairs ordered (row, hi, col)."""
        return tl.reshape(
            tl.broadcast_to(tl.reshape(v, (ROWS, 1, COLS)), (ROWS, 2, COLS)), (ROWS * 2 * COLS,),
        )


    @triton.jit
    def expand_quad(v, ROWS: tl.constexpr, COLS: tl.constexpr):
        """Expand [ROWS * COLS] bytes into 2-bit lanes ordered (row, hi2, col)."""
        return tl.reshape(
            tl.broadcast_to(tl.reshape(v, (ROWS, 1, COLS)), (ROWS, 4, COLS)), (ROWS * 4 * COLS,),
        )


    @triton.jit
    def decode_block_wide(q_ptr, block_base, kvalues_ptr, TYPE: tl.constexpr):
        """Decode one whole 256-value ggml block shared by all lanes.

        All lanes of the program address the same block (lane = value index),
        so every packed field loads as one contiguous vector and the lane
        mapping is done with reshape/broadcast instead of scattered loads.
        """
        w = tl.arange(0, 256)
        if TYPE == 0:  # Q2_K
            scale = expand_group(tl.load(q_ptr + block_base + tl.arange(0, 16)), 16, 16)
            qs = expand_quad(tl.load(q_ptr + block_base + 16 + tl.arange(0, 64)), 2, 32)
            quant = (qs >> (((w // 32) % 4) * 2)) & 3
            d = load_f16(q_ptr, block_base + 80)
            dmin = load_f16(q_ptr, block_base + 82)
            return (d * (scale & 15).to(tl.float32) * quant.to(tl.float32)
                    - dmin * (scale >> 4).to(tl.float32))

        elif TYPE == 1:  # Q3_K
            h = expand_along(tl.load(q_ptr + block_base + tl.arange(0, 32)), 8, 32)
            q_high = (((h >> (w // 32)) & 1) ^ 1).to(tl.int32)
            qs = expand_quad(tl.load(q_ptr + block_base + 32 + tl.arange(0, 64)), 2, 32)
            q_low = (qs >> (((w // 32) % 4) * 2)) & 3
            quant = q_low.to(tl.int32) - (q_high << 2)

            sg16 = tl.arange(0, 16)
            low = expand_along(tl.load(q_ptr + block_base + 96 + tl.arange(0, 8)), 2, 8)
            low = (low >> ((sg16 // 8) * 4)) & 15
            high = expand_along(tl.load(q_ptr + block_base + 104 + tl.arange(0, 4)), 4, 4)
            high = (high >> ((sg16 // 4) * 2)) & 3
            scale = expand_group((low | (high << 4)).to(tl.int32) - 32, 16, 16)
            d = load_f16(q_ptr, block_base + 108)
            return d * scale.to(tl.float32) * quant.to(tl.float32)

        elif TYPE == 2 or TYPE == 3:  # Q4_K / Q5_K
            first = expand_along(tl.load(q_ptr + block_base + 4 + tl.arange(0, 4)), 2, 4)
            middle = expand_along(tl.load(q_ptr + block_base + 8 + tl.arange(0, 4)), 2, 4)
            shared = expand_along(tl.load(q_ptr + block_base + 12 + tl.arange(0, 4)), 2, 4)
            sg = tl.arange(0, 8)
            scale8 = tl.where(
                sg < 4,
                first & 63,
                (shared & 15) | ((first >> 2) & 48),
            )
            minimum8 = tl.where(
                sg < 4,
                middle & 63,
                (shared >> 4) | ((middle >> 2) & 48),
            )
            scale = expand_group(scale8, 8, 32)
            minimum = expand_group(minimum8, 8, 32)

            data_offset: tl.constexpr = 16 if TYPE == 2 else 48
            qs = expand_pairs(
                tl.load(q_ptr + block_base + data_offset + tl.arange(0, 128)), 4, 32,
            )
            quant = ((qs >> (((w // 32) % 2) * 4)) & 15).to(tl.int32)
            if TYPE == 3:
                h = expand_along(tl.load(q_ptr + block_base + 16 + tl.arange(0, 32)), 8, 32)
                quant |= (((h >> (w // 32)) & 1) << 4).to(tl.int32)
            d = load_f16(q_ptr, block_base)
            dmin = load_f16(q_ptr, block_base + 2)
            return (d * scale.to(tl.float32) * quant.to(tl.float32)
                    - dmin * minimum.to(tl.float32))

        elif TYPE == 4:  # Q6_K
            ql = expand_pairs(tl.load(q_ptr + block_base + tl.arange(0, 128)), 2, 64)
            q_low = (ql >> (((w % 128) // 64) * 4)) & 15
            qh = expand_quad(tl.load(q_ptr + block_base + 128 + tl.arange(0, 64)), 2, 32)
            q_high = (qh >> (((w % 128) // 32) * 2)) & 3
            quant = (q_low | (q_high << 4)).to(tl.int32) - 32
            scale = expand_group(
                tl.load(q_ptr + block_base + 192 + tl.arange(0, 16)), 16, 16,
            ).to(tl.int8).to(tl.int32)
            d = load_f16(q_ptr, block_base + 208)
            return d * scale.to(tl.float32) * quant.to(tl.float32)

        else:  # IQ4_XS
            scales_h = load_u16(q_ptr, block_base + 2)
            sl = expand_group(tl.load(q_ptr + block_base + 4 + tl.arange(0, 4)), 4, 2)
            sg = tl.arange(0, 8)
            low = (sl >> ((sg % 2) * 4)) & 15
            high = (scales_h >> (sg * 2)) & 3
            scale = expand_group((low | (high << 4)).to(tl.int32) - 32, 8, 32)
            qs = expand_pairs(tl.load(q_ptr + block_base + 8 + tl.arange(0, 128)), 8, 16)
            index = ((qs >> (((w % 32) // 16) * 4)) & 15).to(tl.int32)
            d = load_f16(q_ptr, block_base)
            return d * scale.to(tl.float32) * tl.load(kvalues_ptr + index).to(tl.float32)


    @triton.jit
    def _dequant_kernel(
        q_ptr, out_ptr, kvalues_ptr, n_values,
        VALUES_PER_BLOCK, TYPE_SIZE,
        BLOCK: tl.constexpr, TYPE: tl.constexpr, OUT_DTYPE: tl.constexpr,
        USE_WIDE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_values
        safe = tl.where(mask, offsets, 0)
        if USE_WIDE:
            value = decode_block_wide(
                q_ptr, tl.program_id(0) * TYPE_SIZE, kvalues_ptr, TYPE=TYPE,
            )
        else:
            value = decode_block_value(
                q_ptr, (safe // VALUES_PER_BLOCK) * TYPE_SIZE,
                safe % VALUES_PER_BLOCK, kvalues_ptr, TYPE=TYPE,
            )
        tl.store(out_ptr + offsets, value.to(OUT_DTYPE), mask=mask)


def triton_in_capture_usable():
    """Whether Triton kernels may launch inside a CUDA graph capture.

    Triton resolves every kernel pointer with ``cuPointerGetAttribute``, which
    rejects the graph-pool addresses ComfyUI's cudaMallocAsync allocator hands
    out. Under the native allocator the same kernels run inside captures
    without issue, and they are by far the fastest path for a captured decode
    step, so capture only falls back to dequantization when it has to.
    """
    global _triton_capture_ok
    if _triton_capture_ok is None:
        try:
            _triton_capture_ok = torch.cuda.memory.get_allocator_backend() != "cudaMallocAsync"
        except Exception:
            _triton_capture_ok = True
    return _triton_capture_ok


def cuda_dequantize(blocks, qtype, dtype=None):
    """Fused single-pass dequantization on CUDA; falls back to the block ops."""
    if triton is None or not blocks.is_cuda or qtype not in CUDA_QTYPES:
        return None
    if torch.cuda.is_current_stream_capturing() and not triton_in_capture_usable():
        # The fused kernel cannot run on capture-allocated memory (see above);
        # the block ops are capture-safe.
        return None
    out_dtype = dtype if dtype in _TL_DTYPES else (torch.float16 if dtype is None else None)
    if out_dtype is None:
        return None
    if not blocks.is_contiguous():
        blocks = blocks.contiguous()
    flat = blocks.reshape(-1)
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    out = torch.empty(flat.numel() // type_size * block_size, device=flat.device, dtype=out_dtype)
    tl_dtype = getattr(tl, _TL_DTYPES[out_dtype])
    _dequant_kernel[(triton.cdiv(out.numel(), 256),)](
        flat, out, get_kvalues(flat.device), out.numel(),
        block_size, type_size,
        BLOCK=256, TYPE=CUDA_QTYPES[qtype], OUT_DTYPE=tl_dtype,
        USE_WIDE=CUDA_QTYPES[qtype] in WIDE_FORMATS,
        num_warps=_DEQUANT_NUM_WARPS,
    )
    return out.view(-1, block_size)

def _with_cuda_dequantize(qtype, fn):
    def dequantize_blocks(blocks, block_size, type_size, dtype=None):
        result = cuda_dequantize(blocks, qtype, dtype)
        if result is not None:
            return result
        return fn(blocks, block_size, type_size, dtype)
    return dequantize_blocks

_raw_dequantize_functions = {
    gguf.GGMLQuantizationType.BF16: dequantize_blocks_BF16,
    gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
    gguf.GGMLQuantizationType.Q5_1: dequantize_blocks_Q5_1,
    gguf.GGMLQuantizationType.Q5_0: dequantize_blocks_Q5_0,
    gguf.GGMLQuantizationType.Q4_1: dequantize_blocks_Q4_1,
    gguf.GGMLQuantizationType.Q4_0: dequantize_blocks_Q4_0,
    gguf.GGMLQuantizationType.Q6_K: dequantize_blocks_Q6_K,
    gguf.GGMLQuantizationType.Q5_K: dequantize_blocks_Q5_K,
    gguf.GGMLQuantizationType.Q4_K: dequantize_blocks_Q4_K,
    gguf.GGMLQuantizationType.Q3_K: dequantize_blocks_Q3_K,
    gguf.GGMLQuantizationType.Q2_K: dequantize_blocks_Q2_K,
    gguf.GGMLQuantizationType.IQ4_NL: dequantize_blocks_IQ4_NL,
    gguf.GGMLQuantizationType.IQ4_XS: dequantize_blocks_IQ4_XS,
}

dequantize_functions = {
    qtype: _with_cuda_dequantize(qtype, fn) if qtype in CUDA_QTYPES else fn
    for qtype, fn in _raw_dequantize_functions.items()
}
