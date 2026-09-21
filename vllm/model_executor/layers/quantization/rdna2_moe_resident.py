# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Load-time preparation for the opt-in RDNA2 resident W4A16 MoE path."""

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_rdna2 import (  # noqa: E501
    CompressedTensorsWNA16RDNA2MoEMethod,
)
from vllm.model_executor.utils import replace_parameter


def pack_sequential_weight(weight: torch.Tensor) -> torch.Tensor:
    """Convert sequential uint8 W4A16 storage to native RDNA2 words."""
    if weight.dtype != torch.uint8 or not weight.is_contiguous():
        raise ValueError("resident RDNA2 weights must be contiguous uint8 tensors")
    if weight.shape[-1] % 4:
        raise ValueError("resident RDNA2 weights need four packed bytes per int32")
    return weight.view(torch.int32).transpose(1, 2).contiguous()


def prepare_resident_layer(layer, group_size: int) -> None:
    """Replace sequential MoE parameters with one native resident layout."""
    if (
        layer.w13_scales.dtype != torch.float16
        or layer.w2_scales.dtype != torch.float16
    ):
        raise ValueError("resident RDNA2 scales must use torch.float16")

    w13 = pack_sequential_weight(layer.w13_qweight)
    w2 = pack_sequential_weight(layer.w2_qweight)
    w13_scale = layer.w13_scales.transpose(1, 2).contiguous()
    w2_scale = layer.w2_scales.transpose(1, 2).contiguous()

    replace_parameter(layer, "w13_qweight", w13)
    replace_parameter(layer, "w2_qweight", w2)
    replace_parameter(layer, "w13_scales", w13_scale)
    replace_parameter(layer, "w2_scales", w2_scale)

    # Reuse the donor/native post-load implementation exactly once.  The shim
    # keeps native-only fields out of the module's parameter/state-dict names.
    native_layer = SimpleNamespace(
        w13_weight_packed=layer.w13_qweight,
        w2_weight_packed=layer.w2_qweight,
        w13_weight_scale=layer.w13_scales,
        w2_weight_scale=layer.w2_scales,
    )
    native_method = object.__new__(CompressedTensorsWNA16RDNA2MoEMethod)
    native_method.group_size = group_size
    native_method.process_weights_after_loading(native_layer)
    layer._rdna2_resident = native_layer


def resident_op_available(device: torch.device) -> bool:
    """Return whether the RDNA2 fused W4A16 operator is available."""
    if device.type != "cuda" or not hasattr(torch.ops, "_rocm_C"):
        return False
    if not hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna2"):
        return False
    return (
        torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] == "gfx1030"
    )


def apply_resident(layer, x, topk_weights, topk_ids):
    """Dispatch through the existing fused RDNA2 routed-MoE helper."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_rdna2 import (  # noqa: E501
        _rdna2_fused_moe,
    )

    activation = layer.activation
    if not isinstance(activation, MoEActivation):
        activation = MoEActivation.from_str(activation)
    return _rdna2_fused_moe(
        x,
        topk_weights,
        topk_ids,
        layer=layer._rdna2_resident,
        activation=activation,
        apply_router_weight_on_input=layer.apply_router_weight_on_input,
        global_num_experts=layer.global_num_experts,
        expert_map=layer.expert_map,
    )
