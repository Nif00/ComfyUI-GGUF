"""Small-batch CUDA matmul kernels for GGML quantized weights.

These kernels target autoregressive decode (up to 16 activation rows, e.g.
speculative-draft verification) and consume packed ggml bytes directly without
materializing a dense weight. The dequantize-then-cuBLAS path remains
preferable for prompt prefill.

Single-row K-quant decode uses the dp4a kernel: the activation is quantized to
int8 per 32 values (ggml's Q8_1 scheme, as llama.cpp's CUDA matvec does) and
packed weight words are multiplied four values per instruction.
"""

import torch
import gguf
import comfy.model_management

try:
    import triton
    import triton.language as tl
except ImportError:  # Optional acceleration; callers retain the old path.
    triton = None
    tl = None

from . import dequant as _dequant
from .dequant import CUDA_QTYPES

_TL_OUT_DTYPES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}

_MATVEC_NUM_WARPS = 1


# qtype -> (kernel format id, rows per program, warps)
_DP4A_QTYPES = {
    gguf.GGMLQuantizationType.Q3_K: (1, 32, 4),
    gguf.GGMLQuantizationType.Q4_K: (2, 8, 2),
    gguf.GGMLQuantizationType.Q5_K: (3, 8, 2),
    gguf.GGMLQuantizationType.Q6_K: (4, 8, 2),
}
_DP4A_EAGER_MIN_BYTES = 256 * 1024 * 1024


def k_quant_matmul_available():
    return triton is not None and torch.cuda.is_available()


def _dp4a_usable(device):
    # inline PTX: NVIDIA only
    return torch.version.hip is None and torch.cuda.get_device_capability(device) >= (6, 1)


if triton is not None:
    @triton.jit
    def _inverse_v_head_index(index, start, num_k, num_v_per_k, head_dim):
        relative = index - start
        k = relative // (num_v_per_k * head_dim)
        rem = relative % (num_v_per_k * head_dim)
        v = rem // head_dim
        d = rem % head_dim
        source = start + (v * num_k + k) * head_dim + d
        return tl.where(index >= start, source, index)


    @triton.jit
    def _k_quant_matvec_kernel(
        x_addr, q_addr, bias_addr, out_addr, kvalues_addr,
        K, N, ROWS, X_ROW_STRIDE, ROW_START, COL_START,
        NUM_K, NUM_V_PER_K, HEAD_DIM, Q_BLOCK_SIZE, Q_TYPE_SIZE,
        BLOCK_M: tl.constexpr,
        TYPE: tl.constexpr,
        USE_WIDE: tl.constexpr,
        HAS_ROW_MAP: tl.constexpr, HAS_COL_MAP: tl.constexpr, HAS_BIAS: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ):
        x_ptr = x_addr.to(tl.pointer_type(OUT_DTYPE))
        q_ptr = q_addr.to(tl.pointer_type(tl.uint8))
        bias_ptr = bias_addr.to(tl.pointer_type(OUT_DTYPE))
        out_ptr = out_addr.to(tl.pointer_type(OUT_DTYPE))
        kvalues_ptr = kvalues_addr.to(tl.pointer_type(tl.int8))
        col = tl.program_id(0)
        rows_idx = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows_idx < ROWS

        source_row = col
        if HAS_ROW_MAP:
            source_row = _inverse_v_head_index(col, ROW_START, NUM_K, NUM_V_PER_K, HEAD_DIM)

        # Each program accumulates one output column over the full K dimension
        # and writes it exactly once: no zero-fill, no atomics, no output copy.
        # Keep the lane axis alive so the cross-lane reduction runs once.
        acc = tl.zeros((BLOCK_M, 256), tl.float32)
        for kb in tl.range(0, K // 256):
            logical_col = kb * 256 + tl.arange(0, 256)
            if USE_WIDE:
                weight = _dequant.decode_block_wide(
                    q_ptr,
                    (source_row * (K // Q_BLOCK_SIZE) + kb) * Q_TYPE_SIZE,
                    kvalues_ptr, TYPE=TYPE,
                )
            else:
                source_col = logical_col
                if HAS_COL_MAP:
                    source_col = _inverse_v_head_index(
                        logical_col, COL_START, NUM_K, NUM_V_PER_K, HEAD_DIM,
                    )
                block_base = (source_row * (K // Q_BLOCK_SIZE) + source_col // Q_BLOCK_SIZE) * Q_TYPE_SIZE
                weight = _dequant.decode_block_value(
                    q_ptr, block_base, source_col % Q_BLOCK_SIZE, kvalues_ptr, TYPE=TYPE,
                )
            activation = tl.load(
                x_ptr + rows_idx[:, None] * X_ROW_STRIDE + logical_col[None, :],
                mask=row_mask[:, None], other=0.0,
            )
            acc += weight[None, :] * activation

        out_row = tl.sum(acc, axis=1)
        if HAS_BIAS:
            out_row += tl.load(bias_ptr + col)
        tl.store(out_ptr + rows_idx * N + col, out_row.to(OUT_DTYPE), mask=row_mask)



    @triton.jit
    def _forward_v_head_index(index, start, num_k, num_v_per_k, head_dim):
        # inverse of _inverse_v_head_index: stored (tiled) position -> logical position
        relative = index - start
        v = relative // (num_k * head_dim)
        k = (relative % (num_k * head_dim)) // head_dim
        d = relative % head_dim
        return tl.where(index >= start, start + (k * num_v_per_k + v) * head_dim + d, index)


    @triton.jit
    def _quantize_q8_kernel(
        x_addr, xq_addr, dx_addr, xsum_addr, xs16_addr,
        COL_START, NUM_K, NUM_V_PER_K, HEAD_DIM,
        X_DTYPE: tl.constexpr, HAS_COL_MAP: tl.constexpr,
    ):
        # 256 activations per program, in stored weight-column order: int8 per
        # 32 values plus per-32 float sums and per-16 int8 sums for the offsets
        x_ptr = x_addr.to(tl.pointer_type(X_DTYPE))
        kb = tl.program_id(0)
        index = kb * 256 + tl.arange(0, 8)[:, None] * 32 + tl.arange(0, 32)[None, :]
        source = index
        if HAS_COL_MAP:
            source = _forward_v_head_index(index, COL_START, NUM_K, NUM_V_PER_K, HEAD_DIM)
        x = tl.load(x_ptr + source).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=1)
        q = tl.extra.cuda.libdevice.rint(x * tl.where(amax > 0, 127.0 / amax, 0.0)[:, None]).to(tl.int32)
        tl.store(xq_addr.to(tl.pointer_type(tl.int8)) + index, q.to(tl.int8))
        tl.store(dx_addr.to(tl.pointer_type(tl.float32)) + kb * 8 + tl.arange(0, 8), amax / 127.0)
        tl.store(xsum_addr.to(tl.pointer_type(tl.float32)) + kb * 8 + tl.arange(0, 8), tl.sum(x, axis=1))
        tl.store(xs16_addr.to(tl.pointer_type(tl.int32)) + kb * 16 + tl.arange(0, 16), tl.sum(tl.reshape(q, (16, 16)), axis=1))


    @triton.jit
    def _dp4a(a, b):
        return tl.inline_asm_elementwise("dp4a.s32.s32 $0, $1, $2, 0;", "=r,r,r", [a, b], dtype=tl.int32, is_pure=True, pack=1)


    @triton.jit
    def _word(q32, off, idx, SHIFT: tl.constexpr):
        # 32-bit word `idx` of a block starting at byte `off`, where off % 4 == SHIFT // 8
        if SHIFT == 0:
            return tl.load(q32 + off // 4 + idx)
        a = (off - 2) // 4 + idx
        return ((tl.load(q32 + a) >> 16) & 0xFFFF) | (tl.load(q32 + a + 1) << 16)


    @triton.jit
    def _byte(word, i):
        return (word >> (8 * i)) & 0xFF


    @triton.jit
    def _half(q_ptr, off):
        return tl.load(q_ptr.to(tl.pointer_type(tl.uint16)) + off // 2).to(tl.float16, bitcast=True).to(tl.float32)


    @triton.jit
    def _k4_scale_min(s0, s1, s2, g):
        # ggml get_scale_min_k4 for sub-block g from the three packed scale words
        first = _byte(s0, g % 4)
        middle = _byte(s1, g % 4)
        shared = _byte(s2, g % 4)
        sc = tl.where(g < 4, first & 63, (shared & 15) | ((first >> 2) & 48)).to(tl.float32)
        mn = tl.where(g < 4, middle & 63, (shared >> 4) | ((middle >> 2) & 48)).to(tl.float32)
        return sc, mn


    @triton.jit
    def _q45k_block(q32, x32, dx_ptr, xsum_ptr, off, kb, BLOCK_N: tl.constexpr, TYPE: tl.constexpr):
        # qs word w: chunk c = w // 8 ; low nibbles are values 64c + 4(w%8), high nibbles +32 ; sub-blocks 2c, 2c+1
        w = tl.arange(0, 32)[None, :]
        o2 = off[:, None]
        QS_WORD: tl.constexpr = 4 if TYPE == 2 else 12
        qs = _word(q32, o2, QS_WORD + w, 0)
        lo = qs & 0x0F0F0F0F
        hi = (qs >> 4) & 0x0F0F0F0F
        if TYPE == 3:
            qh = _word(q32, o2, 4 + (w % 8), 0)
            lo |= ((qh >> (2 * (w // 8))) & 0x01010101) << 4
            hi |= ((qh >> (2 * (w // 8) + 1)) & 0x01010101) << 4
        xw = kb * 64 + 16 * (w // 8) + (w % 8)
        dlo = tl.sum(tl.reshape(_dp4a(lo, tl.load(x32 + xw)), (BLOCK_N, 4, 8)), axis=2).to(tl.float32)
        dhi = tl.sum(tl.reshape(_dp4a(hi, tl.load(x32 + xw + 8)), (BLOCK_N, 4, 8)), axis=2).to(tl.float32)
        dm = _word(q32, off, 0, 0)
        d = (dm & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)[:, None]
        dmin = ((dm >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)[:, None]
        s0 = _word(q32, o2, 1, 0)
        s1 = _word(q32, o2, 2, 0)
        s2 = _word(q32, o2, 3, 0)
        ge = 2 * tl.arange(0, 4)[None, :]
        sc_e, mn_e = _k4_scale_min(s0, s1, s2, ge)
        sc_o, mn_o = _k4_scale_min(s0, s1, s2, ge + 1)
        t = d * (sc_e * tl.load(dx_ptr + kb * 8 + ge) * dlo + sc_o * tl.load(dx_ptr + kb * 8 + ge + 1) * dhi) \
            - dmin * (mn_e * tl.load(xsum_ptr + kb * 8 + ge) + mn_o * tl.load(xsum_ptr + kb * 8 + ge + 1))
        return tl.sum(t, axis=1)


    @triton.jit
    def _q3k_block(q32, q_ptr, x32, dx_ptr, xs16_ptr, off, kb, BLOCK_N: tl.constexpr, SHIFT: tl.constexpr):
        # tile [BN, j, h, i]: qs word 8 + 8h + i, hmask word i ; values 128h + 32j + 4i ; sub-block 8h + 2j + i//4
        j = tl.arange(0, 4)[None, :, None, None]
        h = tl.arange(0, 2)[None, None, :, None]
        i = tl.arange(0, 8)[None, None, None, :]
        o4 = off[:, None, None, None]
        qs = _word(q32, o4, 8 + 8 * h + i, SHIFT)
        hm = _word(q32, o4, i, SHIFT)
        u = ((qs >> (2 * j)) & 0x03030303) | (((hm >> (4 * h + j)) & 0x01010101) << 2)
        dot = tl.sum(tl.reshape(_dp4a(u, tl.load(x32 + kb * 64 + 32 * h + 8 * j + i)), (BLOCK_N, 4, 2, 2, 4)), axis=4)
        a = tl.arange(0, 2)[None, None, None, :]
        g = 8 * h + 2 * j + a
        s0 = _word(q32, o4, 24, SHIFT)
        s1 = _word(q32, o4, 25, SHIFT)
        s2 = _word(q32, o4, 26, SHIFT)
        li = 2 * j + a
        low = (_byte(tl.where(li < 4, s0, s1), li % 4) >> (4 * h)) & 15
        high = (_byte(s2, li % 4) >> (2 * (2 * h + j // 2))) & 3
        sc = ((low | (high << 4)) - 32).to(tl.float32)
        corr = (dot - 4 * tl.load(xs16_ptr + kb * 16 + g)).to(tl.float32)
        t = sc * tl.load(dx_ptr + kb * 8 + 4 * h + j) * corr
        return _half(q_ptr, off + 108) * tl.sum(tl.sum(tl.sum(t, axis=3), axis=2), axis=1)


    @triton.jit
    def _q6k_block(q32, q_ptr, x32, dx_ptr, xs16_ptr, off, kb, BLOCK_N: tl.constexpr, SHIFT: tl.constexpr):
        # tile [BN, h, m, i]: ql word 16h + 8m + i, qh word 32 + 8h + i ; low nibbles are
        # values 128h + 32m + 4i, high nibbles +64 ; sub-blocks 8h + 2m + i//4 (+4)
        h = tl.arange(0, 2)[None, :, None, None]
        m = tl.arange(0, 2)[None, None, :, None]
        i = tl.arange(0, 8)[None, None, None, :]
        o4 = off[:, None, None, None]
        ql = _word(q32, o4, 16 * h + 8 * m + i, SHIFT)
        qh = _word(q32, o4, 32 + 8 * h + i, SHIFT)
        ulo = (ql & 0x0F0F0F0F) | (((qh >> (2 * m)) & 0x03030303) << 4)
        uhi = ((ql >> 4) & 0x0F0F0F0F) | (((qh >> (2 * (2 + m))) & 0x03030303) << 4)
        xw = kb * 64 + 32 * h + 8 * m + i
        dlo = tl.sum(tl.reshape(_dp4a(ulo, tl.load(x32 + xw)), (BLOCK_N, 2, 2, 2, 4)), axis=4)
        dhi = tl.sum(tl.reshape(_dp4a(uhi, tl.load(x32 + xw + 16)), (BLOCK_N, 2, 2, 2, 4)), axis=4)
        a = tl.arange(0, 2)[None, None, None, :]
        g = 8 * h + 2 * m + a
        sc_lo = _byte(tl.where(h == 0, _word(q32, o4, 48, SHIFT), _word(q32, o4, 50, SHIFT)), 2 * m + a)
        sc_hi = _byte(tl.where(h == 0, _word(q32, o4, 49, SHIFT), _word(q32, o4, 51, SHIFT)), 2 * m + a)
        clo = (dlo - 32 * tl.load(xs16_ptr + kb * 16 + g)).to(tl.float32)
        chi = (dhi - 32 * tl.load(xs16_ptr + kb * 16 + g + 4)).to(tl.float32)
        t = sc_lo.to(tl.int8).to(tl.float32) * tl.load(dx_ptr + kb * 8 + g // 2) * clo \
            + sc_hi.to(tl.int8).to(tl.float32) * tl.load(dx_ptr + kb * 8 + (g + 4) // 2) * chi
        return _half(q_ptr, off + 208) * tl.sum(tl.sum(tl.sum(t, axis=3), axis=2), axis=1)


    @triton.jit
    def _k_quant_gemv_kernel(
        xq_addr, dx_addr, xsum_addr, xs16_addr, q_addr, bias_addr, out_addr,
        N, K, ROW_START, NUM_K, NUM_V_PER_K, HEAD_DIM,
        TYPE: tl.constexpr, TYPE_SIZE: tl.constexpr, BLOCK_N: tl.constexpr,
        HAS_ROW_MAP: tl.constexpr, HAS_BIAS: tl.constexpr, OUT_DTYPE: tl.constexpr,
    ):
        q_ptr = q_addr.to(tl.pointer_type(tl.uint8))
        q32 = q_addr.to(tl.pointer_type(tl.int32))
        x32 = xq_addr.to(tl.pointer_type(tl.int32))
        dx_ptr = dx_addr.to(tl.pointer_type(tl.float32))
        xsum_ptr = xsum_addr.to(tl.pointer_type(tl.float32))
        xs16_ptr = xs16_addr.to(tl.pointer_type(tl.int32))
        n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n < N
        row = tl.where(n_mask, n, 0)
        if HAS_ROW_MAP:
            row = _inverse_v_head_index(row, ROW_START, NUM_K, NUM_V_PER_K, HEAD_DIM)
        nb = K // 256
        row_off = row.to(tl.int64) * (nb * TYPE_SIZE)
        acc = tl.zeros((BLOCK_N,), tl.float32)
        if TYPE == 2 or TYPE == 3:
            for kb in range(nb):
                acc += _q45k_block(q32, x32, dx_ptr, xsum_ptr, row_off + kb * TYPE_SIZE, kb, BLOCK_N, TYPE)
        else:
            # odd superblocks of these sizes start 2 bytes past a word boundary
            for kp in range(nb // 2):
                off = row_off + 2 * kp * TYPE_SIZE
                if TYPE == 1:
                    acc += _q3k_block(q32, q_ptr, x32, dx_ptr, xs16_ptr, off, 2 * kp, BLOCK_N, 0)
                    acc += _q3k_block(q32, q_ptr, x32, dx_ptr, xs16_ptr, off + TYPE_SIZE, 2 * kp + 1, BLOCK_N, 16)
                else:
                    acc += _q6k_block(q32, q_ptr, x32, dx_ptr, xs16_ptr, off, 2 * kp, BLOCK_N, 0)
                    acc += _q6k_block(q32, q_ptr, x32, dx_ptr, xs16_ptr, off + TYPE_SIZE, 2 * kp + 1, BLOCK_N, 16)
        if HAS_BIAS:
            acc += tl.load(bias_addr.to(tl.pointer_type(OUT_DTYPE)) + n, mask=n_mask, other=0.0).to(tl.float32)
        tl.store(out_addr.to(tl.pointer_type(OUT_DTYPE)) + n, acc.to(OUT_DTYPE), mask=n_mask)


def can_use_k_quant_matmul(qtype, shape, input):
    return (
        k_quant_matmul_available()
        and input.is_cuda
        and qtype in CUDA_QTYPES
        and len(shape) == 2
        and input.shape[-1] % 256 == 0
        and input.numel() // input.shape[-1] <= (16 if CUDA_QTYPES[qtype] in _dequant.WIDE_FORMATS else 8)
    )


def pin_host_weight(weight):
    """Page-lock a host-resident weight that gets streamed to the device.

    ComfyUI's memory plan parks part of the model in host memory and the
    fused kernel then re-reads those packed weights on every forward. Pageable
    transfers block the host for their whole duration and run at a fraction of
    the pinned bandwidth, so each weight is registered once through ComfyUI's
    own bookkeeping (``comfy.model_management.pin_memory``) and later transfers
    become genuinely asynchronous. Returns ``weight`` for chaining.
    """
    if getattr(weight, "_ggml_host_pinned", None) is not None or not isinstance(weight, torch.Tensor):
        return weight
    if getattr(weight, "_ggml_mmap_backed", False):
        # Read-only file mappings cannot be registered, and the failed attempt
        # queues an async CUDA error; remember the decision.
        weight._ggml_host_pinned = False
        return weight
    view = weight if type(weight) is torch.Tensor else weight.as_subclass(torch.Tensor)
    pinned = False
    if view.device.type == "cpu" and comfy.model_management.pin_memory(view):
        weight._ggml_host_pinned_view = view
        pinned = True
    weight._ggml_host_pinned = pinned
    return weight


def unpin_host_weight(weight):
    """Drop the page-lock registration created for ``weight``, if any."""
    view = getattr(weight, "_ggml_host_pinned_view", None)
    if view is not None:
        comfy.model_management.unpin_memory(view)
        weight._ggml_host_pinned_view = None
        weight._ggml_host_pinned = False


def k_quant_matmul(input, qdata, qtype, shape, bias=None, postprocess=()):
    """Compute ``input @ weight.T`` from packed ggml bytes on ``input.device``."""
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    out_features, in_features = map(int, shape)
    rows = input.numel() // in_features

    row_map = col_map = None
    if postprocess:
        if len(postprocess) != 1 or postprocess[0][0] != "qwen35_inverse_v_heads":
            raise ValueError(f"Unsupported quantized matmul postprocess: {postprocess!r}")
        _, dim, start, num_k, num_v_per_k, head_dim = postprocess[0]
        if dim == 0:
            row_map = (start, num_k, num_v_per_k, head_dim)
        elif dim == 1:
            col_map = (start, num_k, num_v_per_k, head_dim)
        else:
            raise ValueError(f"Unsupported Qwen3.5 postprocess dimension: {dim}")

    mapping = row_map or col_map or (0, 1, 1, 1)
    x = input.reshape(rows, in_features)
    if x.stride(1) != 1:
        x = x.contiguous()

    direct_dtype = input.dtype in _TL_OUT_DTYPES
    out = torch.empty(
        (rows, out_features), device=input.device,
        dtype=input.dtype if direct_dtype else torch.float32,
    )
    if not direct_dtype:
        x = x.float()
    if bias is not None:
        bias = bias.to(device=input.device, dtype=out.dtype).reshape(-1).contiguous()

    out_dtype = getattr(tl, _TL_OUT_DTYPES[out.dtype])
    # Addresses are passed as integers: Triton's launcher rejects tensors
    # allocated inside a cudaMallocAsync graph capture.
    dp4a = _DP4A_QTYPES.get(qtype)
    # dp4a costs a second launch for the activation quantization. Eager decode
    # is launch-bound, so it only pays off inside a CUDA graph or for weights
    # large enough that GPU time dominates (a vocab-sized output projection).
    worth_second_launch = torch.cuda.is_current_stream_capturing() or qdata.numel() >= _DP4A_EAGER_MIN_BYTES
    if dp4a is not None and rows == 1 and worth_second_launch and (type_size % 4 == 0 or in_features % 512 == 0) and _dp4a_usable(input.device):
        type_id, block_n, num_warps = dp4a
        # int8 activations, then per-32 scales, per-32 sums and per-16 int sums
        scratch = torch.empty((in_features * 3 // 2,), device=input.device, dtype=torch.uint8)
        xq = scratch.data_ptr()
        dx, xsum, xs16 = xq + in_features, xq + in_features * 9 // 8, xq + in_features * 5 // 4
        col = col_map or (0, 1, 1, 1)
        _quantize_q8_kernel[(in_features // 256,)](
            x.data_ptr(), xq, dx, xsum, xs16, col[0], col[1], col[2], col[3],
            X_DTYPE=out_dtype, HAS_COL_MAP=col_map is not None, num_warps=2,
        )
        row = row_map or (0, 1, 1, 1)
        _k_quant_gemv_kernel[(triton.cdiv(out_features, block_n),)](
            xq, dx, xsum, xs16, qdata.data_ptr(), (bias if bias is not None else out).data_ptr(), out.data_ptr(),
            out_features, in_features, row[0], row[1], row[2], row[3],
            TYPE=type_id, TYPE_SIZE=type_size, BLOCK_N=block_n,
            HAS_ROW_MAP=row_map is not None, HAS_BIAS=bias is not None, OUT_DTYPE=out_dtype,
            num_warps=num_warps,
        )
        out = out.reshape(*input.shape[:-1], out_features)
        return out if direct_dtype else out.to(input.dtype)

    block_m = min(max(triton.next_power_of_2(rows), 1), 8)
    qtype_id = CUDA_QTYPES[qtype]
    _k_quant_matvec_kernel[(out_features, triton.cdiv(rows, block_m))](
        x.data_ptr(), qdata.data_ptr(), (bias if bias is not None else out).data_ptr(), out.data_ptr(),
        _dequant.get_kvalues(input.device).data_ptr(),
        K=in_features, N=out_features, ROWS=rows, X_ROW_STRIDE=x.stride(0),
        ROW_START=mapping[0], COL_START=mapping[0],
        NUM_K=mapping[1], NUM_V_PER_K=mapping[2], HEAD_DIM=mapping[3],
        Q_BLOCK_SIZE=block_size, Q_TYPE_SIZE=type_size,
        BLOCK_M=block_m,
        TYPE=qtype_id,
        USE_WIDE=qtype_id in _dequant.WIDE_FORMATS and col_map is None,
        HAS_ROW_MAP=row_map is not None,
        HAS_COL_MAP=col_map is not None,
        HAS_BIAS=bias is not None,
        OUT_DTYPE=out_dtype,
        num_warps=_MATVEC_NUM_WARPS,
    )
    out = out.reshape(*input.shape[:-1], out_features)
    return out if direct_dtype else out.to(input.dtype)
