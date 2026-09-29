# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import gguf
import json
import torch
import logging

import comfy.ops
import comfy.lora
import comfy.model_management
from .dequant import dequantize, dequantize_functions, dequantize_tensor, is_quantized
from .quant_matmul import can_use_k_quant_matmul, k_quant_matmul, pin_host_weight

def _valid_compute_dtype(dtype):
    return dtype in {torch.float16, torch.bfloat16, torch.float32, torch.float64}

def _infer_compute_dtype(tensor_type, fallback=None):
    if _valid_compute_dtype(fallback):
        return fallback
    if tensor_type == gguf.GGMLQuantizationType.BF16:
        return torch.bfloat16
    if tensor_type == gguf.GGMLQuantizationType.F32:
        return torch.float32
    return torch.float16

def _to_device_arg(args, kwargs):
    """Parse the device requested by a ``Tensor.to`` call (``args`` excludes self)."""
    device = kwargs.get("device")
    if device is not None:
        return device
    for arg in args:
        if isinstance(arg, (torch.dtype, bool)):
            continue
        if isinstance(arg, torch.Tensor):
            return arg.device
        if isinstance(arg, (torch.device, str)):
            return arg
    return None

def _to_requested_dtype(args, kwargs):
    """Parse the dtype requested by a ``Tensor.to`` call (``args`` excludes self)."""
    dtype = kwargs.get("dtype")
    if dtype is not None:
        return dtype
    for arg in args:
        if isinstance(arg, torch.Tensor):
            return arg.dtype
        if isinstance(arg, torch.dtype):
            return arg
        if isinstance(arg, tuple):
            for item in arg:
                if isinstance(item, torch.dtype):
                    return item
    return None

def chained_hasattr(obj, chained_attr):
    probe = obj
    for attr in chained_attr.split('.'):
        if hasattr(probe, attr):
            probe = getattr(probe, attr)
        else:
            return False
    return True

# A bakcward and forward compatible way to get `torch.compiler.disable`.
def get_torch_compiler_disable_decorator():
    def dummy_decorator(*args, **kwargs):
        def noop(x):
            return x
        return noop

    from packaging import version

    if not chained_hasattr(torch, "compiler.disable"):
        logging.info("ComfyUI-GGUF: Torch too old for torch.compile - bypassing")
        return dummy_decorator # torch too old
    elif version.parse(torch.__version__) >= version.parse("2.8"):
        logging.info("ComfyUI-GGUF: Allowing full torch compile")
        return dummy_decorator # torch compile works
    if chained_hasattr(torch, "_dynamo.config.nontraceable_tensor_subclasses"):
        logging.info("ComfyUI-GGUF: Allowing full torch compile (nightly)")
        return dummy_decorator # torch compile works, nightly before 2.8 release
    else:
        logging.info("ComfyUI-GGUF: Partial torch compile only, consider updating pytorch")
        return torch.compiler.disable

torch_compiler_disable = get_torch_compiler_disable_decorator()

class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights
    """
    def __init__(self, *args, tensor_type, tensor_shape, patches=[], compute_dtype=None, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches
        self.compute_dtype = compute_dtype

    def __new__(cls, *args, tensor_type, tensor_shape, patches=[], compute_dtype=None, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    # Value ops that may receive a packed weight from comfy's cast paths
    # (``CastBiasWeightContext``, ``model_management.cast_to``, ...). Packed
    # blocks must be dequantized before they reach any of them.
    _PACKED_WEIGHT_OPS = {
        "linear", "matmul", "mm", "bmm", "embedding",
        "conv1d", "conv2d", "conv3d",
        "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    }

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        name = getattr(func, "__name__", "")
        if name in cls._PACKED_WEIGHT_OPS:
            operands = list(args) + list(kwargs.values())
            if any(isinstance(a, GGMLTensor) and is_quantized(a) for a in operands):
                if name == "linear":
                    input_tensor = args[0] if args else kwargs.get("input")
                    weight = args[1] if len(args) > 1 else kwargs.get("weight")
                    # The fused kernel consumes the packed bytes directly and
                    # never materializes a dense weight.
                    if weight.supports_fused_linear(input_tensor):
                        return weight.fused_linear(input_tensor, args[2] if len(args) > 2 else kwargs.get("bias"))
                # activations precede weights in every supported signature
                dtype = next(
                    (a.dtype for a in operands if isinstance(a, torch.Tensor) and a.dtype.is_floating_point),
                    None,
                )

                def dense(tensor):
                    if isinstance(tensor, GGMLTensor) and is_quantized(tensor):
                        return dequantize_weight(tensor, dtype or tensor.dtype)
                    return tensor

                return func(*(dense(a) for a in args), **{k: dense(v) for k, v in kwargs.items()})
        return super().__torch_function__(func, types, args, kwargs)

    def supports_fused_linear(self, input):
        return (
            self.device == input.device
            and not getattr(self, "patches", ())
            and can_use_k_quant_matmul(getattr(self, "tensor_type", None), getattr(self, "tensor_shape", ()), input)
        )

    def fused_linear(self, input, bias=None):
        return k_quant_matmul(
            input, self.as_subclass(torch.Tensor), self.tensor_type, self.tensor_shape,
            bias, getattr(self, "gguf_postprocess", ()),
        )

    def _copy_meta(self, new):
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        new.compute_dtype = getattr(self, "compute_dtype", None)
        new.gguf_postprocess = getattr(self, "gguf_postprocess", ())
        new.native_matmul = getattr(self, "native_matmul", False)
        return new

    def to(self, *args, **kwargs):
        target_dtype = _to_requested_dtype(args, kwargs)
        if is_quantized(self) and target_dtype is not None:
            # Converting packed quant blocks as values would corrupt them, so a
            # dtype request never reaches the storage.
            shape = getattr(self, "tensor_shape", ())
            if len(shape) == 2 and _valid_compute_dtype(target_dtype) and not getattr(self, "patches", ()):
                # Linear weights are dtype-agnostic while packed: record the
                # requested compute dtype and let ``__torch_function__`` compute
                # from the packed storage. Materializing here costs the full
                # dense weight (2 GiB for a 9B lm_head) on every MTP verify.
                # Patched weights keep the dense path so the patch machinery
                # (``GGMLModelPatcher`` weight functions) sees real values.
                device = _to_device_arg(args, kwargs)
                new = self if device is None or torch.device(device) == self.device else self._copy_meta(super().to(device=device))
                new.compute_dtype = target_dtype
                return new
            dense = dequantize_weight(self, target_dtype)
            device = _to_device_arg(args, kwargs)
            if device is not None and torch.device(device) != dense.device:
                dense = dense.to(device=device)
            return dense
        return self._copy_meta(super().to(*args, **kwargs))

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def copy_(self, *args, **kwargs):
        # fixes .weight.copy_ in comfy/clip_model/CLIPTextModel
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            logging.warning(f"ignoring 'copy_' on tensor: {e}")

    def new_empty(self, size, *args, **kwargs):
        # Intel Arc fix, ref#50
        new_tensor = super().new_empty(size, *args, **kwargs)
        self._copy_meta(new_tensor).tensor_shape = size
        return new_tensor

    @property
    def dtype(self):
        qtype = getattr(self, "tensor_type", None)
        if qtype in GGMLLayer.torch_compatible_tensor_types:
            return torch.Tensor(self).dtype
        return _infer_compute_dtype(qtype, getattr(self, "compute_dtype", None))

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape

class GGMLLayer(torch.nn.Module):
    """
    This (should) be responsible for de-quantizing on the fly
    """
    comfy_cast_weights = True
    dequant_dtype = None
    patch_dtype = None
    largest_layer = False
    torch_compatible_tensor_types = {None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}

    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return is_quantized(weight) or is_quantized(bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight, bias = state_dict.get(f"{prefix}weight"), state_dict.get(f"{prefix}bias")
        # NOTE: using modified load for linear due to not initializing on creation, see GGMLOps todo
        if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(self, torch.nn.Linear):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # Not strictly required, but fixes embedding shape mismatch. Threshold set in loader.py
        if isinstance(self, torch.nn.Embedding) and self.weight.shape[0] >= (64 * 1024):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        prefix_len = len(prefix)
        for k,v in state_dict.items():
            if k[prefix_len:] == "weight":
                if isinstance(v, GGMLTensor):
                    v.compute_dtype = self._ggml_compute_dtype(v, "weight")
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == "bias" and v is not None:
                if isinstance(v, GGMLTensor):
                    v.compute_dtype = self._ggml_compute_dtype(v, "bias")
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                unexpected_keys.append(k)

        # For Linear layer with missing weight
        if self.weight is None and isinstance(self, torch.nn.Linear):
            v = torch.zeros(self.in_features, self.out_features)
            self.weight = torch.nn.Parameter(v, requires_grad=False)
            missing_keys.append(prefix+"weight")

        # for vram estimation (TODO: less fragile logic?)
        if getattr(self.weight, "is_largest_weight", False):
            self.largest_layer = True

    def _ggml_compute_dtype(self, tensor, param_name):
        if self.dequant_dtype is not None and self.dequant_dtype != "target":
            return self.dequant_dtype
        model_dtype = getattr(self, f"{param_name}_comfy_model_dtype", None)
        return _infer_compute_dtype(getattr(tensor, "tensor_type", None), model_dtype)

    def _save_to_state_dict(self, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)

    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        # This is a fake state dict for vram estimation
        if getattr(self.weight, "native_matmul", False):
            # Native K-quant matmul consumes the packed bytes directly, so its
            # resident size is the physical GGUF storage rather than FP16.
            physical = torch.Tensor(self.weight)
            weight = torch.empty(
                physical.numel(), dtype=physical.dtype, device=torch.device("meta"),
            )
        else:
            weight = torch.zeros_like(self.weight, device=torch.device("meta"))
        destination[prefix + "weight"] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device("meta"))
            destination[prefix + "bias"] = bias

        # Take into account space required for dequantizing the largest tensor
        if self.largest_layer:
            shape = getattr(self.weight, "tensor_shape", self.weight.shape)
            dtype = self.dequant_dtype if self.dequant_dtype and self.dequant_dtype != "target" else torch.float16
            temp = torch.empty(*shape, device=torch.device("meta"), dtype=dtype)
            destination[prefix + "temp.weight"] = temp

        return
        # This would return the dequantized state dict
        destination[prefix + "weight"] = self.get_weight(self.weight)
        if bias is not None:
            destination[prefix + "bias"] = self.get_weight(self.bias)

    def get_weight(self, tensor, dtype):
        if tensor is None:
            return
        patch_dtype = None
        if self.patch_dtype is not None:
            # for testing, may degrade image quality
            patch_dtype = dtype if self.patch_dtype == "target" else self.patch_dtype
        return dequantize_weight(tensor, dtype, self.dequant_dtype, patch_dtype)

    @torch_compiler_disable()
    def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        bias = None
        non_blocking = comfy.model_management.device_supports_non_blocking(device)
        if s.bias is not None:
            bias = s.get_weight(s.bias.to(device), dtype)
            bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking=non_blocking, copy=False)

        weight = s.get_weight(pin_host_weight(s.weight).to(device, non_blocking=non_blocking), dtype)
        weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
        return weight, bias

    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        if self.is_ggml_quantized():
            out = self.forward_ggml_cast_weights(input, *args, **kwargs)
        else:
            out = super().forward_comfy_cast_weights(input, *args, **kwargs)

        # non-ggml forward might still propagate custom tensor class
        if isinstance(out, GGMLTensor):
            out = torch.Tensor(out)
        return out

    def forward_ggml_cast_weights(self, input):
        raise NotImplementedError

class GGMLOps(comfy.ops.manual_cast):
    """
    Dequantize weights on the fly before doing the compute
    """
    class Linear(GGMLLayer, comfy.ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            # TODO: better workaround for reserved memory spike on windows
            # Issue is with `torch.empty` still reserving the full memory for the layer
            # Windows doesn't over-commit memory so without this 24GB+ of pagefile is used
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None
            self.weight_comfy_model_dtype = dtype
            self.bias_comfy_model_dtype = dtype

        def forward_ggml_cast_weights(self, input):
            if self.weight.supports_fused_linear(input):
                return self.weight.fused_linear(input, self.bias)
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(GGMLLayer, comfy.ops.manual_cast.Conv2d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)

    class Embedding(GGMLLayer, comfy.ops.manual_cast.Embedding):
        def forward_ggml_cast_weights(self, input, out_dtype=None):
            weight = self.weight
            qtype = getattr(weight, "tensor_type", None)
            shape = getattr(weight, "tensor_shape", None)
            if (
                is_quantized(weight)
                and qtype in dequantize_functions
                and shape is not None and len(shape) == 2
                and shape[1] % gguf.GGML_QUANT_SIZES[qtype][0] == 0
                and not getattr(weight, "patches", ())
                and not getattr(weight, "gguf_postprocess", ())
            ):
                # Gather the packed rows for the requested ids and dequantize
                # only those; materializing the full vocab table is gigabytes.
                rows, columns = map(int, shape)
                packed = (weight if type(weight) is torch.Tensor else weight.as_subclass(torch.Tensor)).reshape(rows, -1)
                ids = input.reshape(-1).to(packed.device)
                selected = packed.index_select(0, ids).to(input.device, non_blocking=True)
                dtype = out_dtype or _infer_compute_dtype(qtype, weight.compute_dtype)
                result = dequantize(selected, qtype, torch.Size((ids.numel(), columns)), dtype=dtype)
                return result.reshape(*input.shape, columns)
            output_dtype = out_dtype
            if self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16:
                out_dtype = None
            weight, _bias = self.cast_bias_weight(self, device=input.device, dtype=out_dtype)
            return torch.nn.functional.embedding(
                input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
            ).to(dtype=output_dtype)

    class LayerNorm(GGMLLayer, comfy.ops.manual_cast.LayerNorm):
        def forward_ggml_cast_weights(self, input):
            if self.weight is None:
                return super().forward_comfy_cast_weights(input)
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

    class GroupNorm(GGMLLayer, comfy.ops.manual_cast.GroupNorm):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    else:
        return item

def dequantize_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
    """Dense weights for a packed GGUF tensor with any attached patches applied.

    Shared by the layer forwards (via ``GGMLLayer.get_weight``) and by
    ``GGMLTensor.__torch_function__``, which materializes weights handed to
    value ops outside of a layer forward (comfy's cast paths). ``patch_dtype``
    is the weighted-patch compute dtype (``None`` keeps the lora default).
    """
    # consolidate and load patches to GPU in async
    patch_list = []
    key = None
    for patches, key in getattr(tensor, "patches", ()):
        patch_list += move_patch_to_device(patches, tensor.device)

    # dequantize tensor while patches load
    weight = dequantize_tensor(tensor, dtype, dequant_dtype)

    # prevent propagating custom tensor class
    if isinstance(weight, GGMLTensor):
        weight = torch.Tensor(weight)

    if len(patch_list) > 0:
        if patch_dtype is None:
            weight = comfy.lora.calculate_weight(patch_list, weight, key)
        else:
            weight = comfy.lora.calculate_weight(patch_list, weight, key, patch_dtype)
    return weight

def get_gguf_q8_ops(compute_dtype=torch.bfloat16, full_precision_mm=False):
    """
    Factory for an ops class that uses ComfyUI's native mixed_precision_ops INT8 path.
    Weights are kept as INT8 and matmul uses comfy_kitchen's TensorWiseINT8Layout.
    """
    BaseOps = comfy.ops.mixed_precision_ops(
        quant_config={},
        compute_dtype=compute_dtype,
        full_precision_mm=full_precision_mm,
    )

    class GGUFQ8Ops(BaseOps):
        class Linear(BaseOps.Linear):
            def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
                # Lazy init: don't allocate weight here; it will be loaded from state dict
                torch.nn.Module.__init__(self)
                self.factory_kwargs = {"device": device, "dtype": BaseOps._compute_dtype}
                self.in_features = in_features
                self.out_features = out_features
                self.weight = None
                if bias:
                    self.bias = torch.nn.Parameter(torch.empty(out_features, **self.factory_kwargs))
                else:
                    self.register_parameter("bias", None)
                self._orig_shape = (out_features, in_features)
                self.tensor_class = None
                self._full_precision_mm = BaseOps._full_precision_mm
                self._full_precision_mm_config = False

            def _load_from_state_dict(self, *args):
                state_dict, prefix = args[:2]
                weight_key = f"{prefix}weight"
                # Target-size GGUFs combine native Q8_CR weights with standard
                # Q4_0 weights. The mixed-precision loader only understands
                # the former's comfy_quant metadata, so materialize standard
                # GGML weights before handing them to that loader.
                if weight_key in state_dict and f"{prefix}comfy_quant" not in state_dict:
                    weight = state_dict[weight_key]
                    if is_quantized(weight):
                        state_dict[weight_key] = dequantize_tensor(
                            weight,
                            dtype=self.factory_kwargs["dtype"],
                        )
                    elif hasattr(weight, "dequantize"):
                        state_dict[weight_key] = weight.dequantize().to(
                            dtype=self.factory_kwargs["dtype"],
                        )
                return comfy.ops._load_quantized_module(
                    self,
                    torch.nn.Module._load_from_state_dict.__get__(self, type(self)),
                    *args,
                    load_extra_params=True,
                )

    return GGUFQ8Ops


# Retired experimental Q4_PT implementation. No loader or node runtime path
# references this class until a performant W4A16 backend is available.
class RetiredGGUFQ4Ops(comfy.ops.manual_cast):
    """
    Ops class for PyTorch's native compact INT4 GEMM.

    Packed weights are created transiently at invocation time. This keeps
    low-VRAM offloading viable without retaining a second copy of all weights.
    """
    class Linear(torch.nn.Module, comfy.ops.CastWeightBiasOp):
        comfy_cast_weights = True

        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.register_parameter("bias", None)
            self._orig_shape = (out_features, in_features)
            self._group_size = None
            self._pad = 0
            self._orig_in_features = in_features
            self._is_int4 = False

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            weight_key = f"{prefix}weight"
            scale_key = f"{prefix}weight_scale"
            quant_key = f"{prefix}comfy_quant"

            weight = state_dict.pop(weight_key, None)
            scale = state_dict.pop(scale_key, None)
            quant_raw = state_dict.pop(quant_key, None)

            bias_key = f"{prefix}bias"
            bias = state_dict.pop(bias_key, None)
            if quant_raw is None:
                if weight is None:
                    missing_keys.append(weight_key)
                    return
                self.weight = torch.nn.Parameter(torch.Tensor(weight), requires_grad=False)
                if bias is not None:
                    self.bias = torch.nn.Parameter(torch.Tensor(bias), requires_grad=False)
                self._is_int4 = False
                return

            if weight is None or scale is None:
                raise RuntimeError(f"Missing INT4 tensors for {prefix}")

            quant_conf = json.loads(bytes(quant_raw.tolist()).decode("utf-8"))
            if quant_conf.get("format") not in {"int4_compact_gemm", "int4_pytorch"}:
                raise ValueError(f"Unsupported INT4 format for {prefix}")
            self._group_size = quant_conf["group_size"]
            self._pad = quant_conf.get("pad", 0)
            orig_shape = tuple(quant_conf["orig_shape"])
            self._orig_in_features = orig_shape[1]
            self._orig_shape = orig_shape
            self.out_features = orig_shape[0]

            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
            self._is_int4 = True

            if bias is not None:
                self.bias = torch.nn.Parameter(bias, requires_grad=False)
            for key in (weight_key, scale_key, quant_key, bias_key):
                if key in missing_keys:
                    missing_keys.remove(key)

        def forward(self, input):
            if self.weight is None:
                raise RuntimeError("Q4_PT weight was not loaded.")
            if not self._is_int4:
                weight = self.weight.to(device=input.device, dtype=input.dtype)
                bias = self.bias.to(device=input.device, dtype=input.dtype) if self.bias is not None else None
                return torch.nn.functional.linear(input, weight, bias)
            if self.weight_function or self.bias_function:
                raise RuntimeError("Q4_PT does not support weight patches or LoRAs without dequantization.")
            input_shape = input.shape
            input_2d = input.reshape(-1, input_shape[-1])
            if self._pad:
                input_2d = torch.nn.functional.pad(input_2d, (0, self._pad))

            if self._group_size != 64:
                raise RuntimeError(
                    f"Q4_PT requires PyTorch's group-size-64 INT4 operator, got {self._group_size}."
                )
            if input_2d.device.type != "cuda":
                raise RuntimeError("Q4_PT requires a CUDA device.")
            if input_2d.dtype != torch.bfloat16:
                raise RuntimeError(
                    f"Q4_PT requires BF16 activations for PyTorch's native INT4 operator, got {input_2d.dtype}."
                )

            weight = torch.Tensor(
                self.weight.to(device=input.device, dtype=torch.uint8, non_blocking=True)
            )
            scale_and_offset = self.weight_scale.to(
                device=input.device,
                dtype=torch.bfloat16,
                non_blocking=True,
            )
            packed_features = weight.size(-1) * 2
            native_padding = (-packed_features) % 128
            if native_padding:
                # PyTorch's INT4 packer needs K to be a multiple of 128.
                # A zero-valued group and zero input features preserve the
                # original result for K=64 projections such as Krea2's input.
                if native_padding % self._group_size:
                    raise RuntimeError(
                        f"Cannot pad Q4_PT input width {packed_features} to PyTorch's INT4 tile size."
                    )
                input_2d = torch.nn.functional.pad(input_2d, (0, native_padding))
                weight = torch.nn.functional.pad(weight, (0, native_padding // 2))
                scale_and_offset = torch.nn.functional.pad(
                    scale_and_offset,
                    (0, 0, 0, 0, 0, native_padding // self._group_size),
                )

            packed_weight = torch._convert_weight_to_int4pack(weight, 8)
            output = torch._weight_int4pack_mm(
                input_2d,
                packed_weight,
                self._group_size,
                scale_and_offset,
            )
            if self.bias is not None:
                output = output + self.bias.to(device=input.device, dtype=input.dtype)
            return output.reshape(*input_shape[:-1], self.out_features)
