# GGML QuantizedTensor support for ComfyUI DynamicVRAM loading.
import dataclasses
from dataclasses import dataclass

import gguf
import torch

from comfy_kitchen.tensor import (
    BaseLayoutParams,
    QuantizedLayout,
    QuantizedTensor,
    register_layout_class,
    register_layout_op,
)

from .dequant import TORCH_COMPATIBLE_QTYPES, apply_tensor_postprocess, dequantize, dequantize_functions
from .quant_matmul import can_use_k_quant_matmul, k_quant_matmul


@dataclass(frozen=True)
class GGMLLayoutParams(BaseLayoutParams):
    tensor_type: int
    postprocess: tuple = ()
    transposed: bool = False


class GGMLLayout(QuantizedLayout):
    Params = GGMLLayoutParams

    @classmethod
    def quantize(cls, tensor, **kwargs):
        raise NotImplementedError("Quantization to GGML format is not supported")

    @classmethod
    def dequantize(cls, qdata, params):
        return _dequantize(qdata, params, params.orig_dtype)

    @classmethod
    def get_plain_tensors(cls, qtensor):
        return (qtensor._qdata,)

    @classmethod
    def state_dict_tensors(cls, qdata, params):
        return {"weight": qdata}


register_layout_class("GGMLLayout", GGMLLayout)


def _dequantize(qdata, params, dtype):
    qtype = gguf.GGMLQuantizationType(params.tensor_type)
    shape = params.orig_shape[-2:][::-1] if params.transposed else params.orig_shape

    if qtype in TORCH_COMPATIBLE_QTYPES:
        result = qdata.reshape(shape).to(dtype)
    elif qtype not in dequantize_functions:
        result = torch.from_numpy(gguf.quants.dequantize(qdata.cpu().numpy(), qtype)).reshape(shape).to(
            device=qdata.device,
            dtype=dtype,
        )
    else:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
        raw = qdata.reshape(-1).view(torch.uint8)
        blocks = raw.reshape((raw.numel() // type_size, type_size))
        result = dequantize_functions[qtype](blocks, block_size, type_size, dtype).reshape(shape).to(dtype)
    result = apply_tensor_postprocess(result, params.postprocess)
    if params.transposed:
        result = result.t().expand(params.orig_shape)
    return result


def _linear(input, weight, bias=None):
    """``F.linear`` against a stored (untransposed) GGML weight."""
    params = weight._params
    qtype = gguf.GGMLQuantizationType(params.tensor_type)
    if weight._qdata.device == input.device and can_use_k_quant_matmul(qtype, params.orig_shape, input):
        return k_quant_matmul(input, weight._qdata, qtype, params.orig_shape, bias, params.postprocess)
    return torch.nn.functional.linear(input, _dequantize(weight._qdata, params, input.dtype), bias)


def _stored(rhs):
    """The stored weight behind ``W.t()``, the operand F.linear hands to mm."""
    return QuantizedTensor(rhs._qdata, "GGMLLayout", dataclasses.replace(
        rhs._params, orig_shape=rhs._params.orig_shape[-2:][::-1], transposed=False,
    ))


@register_layout_op(torch.ops.aten.t.default, GGMLLayout)
def _handle_t(qt, args, kwargs):
    old = args[0]._params
    return QuantizedTensor(args[0]._qdata, "GGMLLayout", dataclasses.replace(
        old, orig_shape=old.orig_shape[::-1], transposed=not old.transposed,
    ))


@register_layout_op(torch.ops.aten.linear.default, GGMLLayout)
def _handle_linear(qt, args, kwargs):
    input, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    if isinstance(input, QuantizedTensor) or not isinstance(weight, QuantizedTensor) or weight._params.transposed:
        return torch.nn.functional.linear(*[a.dequantize() if isinstance(a, QuantizedTensor) else a for a in (input, weight, bias)])
    return _linear(input, weight, bias)


@register_layout_op(torch.ops.aten.mm.default, GGMLLayout)
def _handle_mm(qt, args, kwargs):
    a, b = args[0], args[1]
    if isinstance(a, QuantizedTensor) or not isinstance(b, QuantizedTensor) or not b._params.transposed:
        return torch.mm(*[t.dequantize() if isinstance(t, QuantizedTensor) else t for t in (a, b)])
    return _linear(a, _stored(b))


@register_layout_op(torch.ops.aten.addmm.default, GGMLLayout)
def _handle_addmm(qt, args, kwargs):
    bias, a, b = args[0], args[1], args[2]
    if isinstance(a, QuantizedTensor) or not isinstance(b, QuantizedTensor) or not b._params.transposed or bias.dim() != 1:
        return torch.addmm(*[t.dequantize() if isinstance(t, QuantizedTensor) else t for t in (bias, a, b)])
    return _linear(a, _stored(b), bias)


@register_layout_op(torch.ops.aten.expand.default, GGMLLayout)
def _handle_expand(qt, args, kwargs):
    # matmul broadcasts W.t() for bmm when a sliced activation cannot be
    # folded to 2D; record the broadcast shape and keep the weight packed.
    rhs, size = args[0], tuple(args[1])
    if not rhs._params.transposed or len(size) != 3 or size[-2:] != rhs._params.orig_shape[-2:]:
        return rhs.dequantize().expand(size)
    return QuantizedTensor(rhs._qdata, "GGMLLayout", dataclasses.replace(rhs._params, orig_shape=size))


@register_layout_op(torch.ops.aten.view.default, GGMLLayout)
def _handle_view(qt, args, kwargs):
    rhs, size = args[0], tuple(args[1])
    if size == tuple(rhs._params.orig_shape):
        return QuantizedTensor(rhs._qdata, "GGMLLayout", rhs._params)
    return rhs.dequantize().view(size)


@register_layout_op(torch.ops.aten.bmm.default, GGMLLayout)
def _handle_bmm(qt, args, kwargs):
    a, b = args[0], args[1]
    if isinstance(a, QuantizedTensor) or not isinstance(b, QuantizedTensor) or not b._params.transposed:
        return torch.bmm(*[t.dequantize() if isinstance(t, QuantizedTensor) else t for t in (a, b)])
    return _linear(a, _stored(b))


@register_layout_op(torch.ops.aten.embedding.default, GGMLLayout)
def _handle_embedding(qt, args, kwargs):
    weight, indices = args[0], args[1]
    params = weight._params
    qtype = gguf.GGMLQuantizationType(params.tensor_type)
    if params.postprocess or params.transposed or qtype not in dequantize_functions:
        return torch.nn.functional.embedding(indices, weight.dequantize())
    # Gather and dequantize only the requested rows; the whole vocab table is
    # gigabytes once dense.
    rows, columns = params.orig_shape
    ids = indices.reshape(-1).to(weight._qdata.device)
    selected = weight._qdata.reshape(rows, -1).index_select(0, ids)
    return dequantize(selected, qtype, torch.Size((ids.numel(), columns)), dtype=params.orig_dtype).reshape(*indices.shape, columns)


def make_quantized(qdata, tensor_type, tensor_shape, orig_dtype=torch.float16, postprocess=()):
    params = GGMLLayoutParams(
        scale=torch.ones((), dtype=torch.float32),
        orig_dtype=orig_dtype,
        orig_shape=tuple(tensor_shape),
        tensor_type=tensor_type.value if not isinstance(tensor_type, int) else tensor_type,
        postprocess=tuple(postprocess),
    )
    return QuantizedTensor(qdata, "GGMLLayout", params)
