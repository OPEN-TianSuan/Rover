"""Optional quantized linear layers."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def quant_method_name(quantization_config) -> str:
    if isinstance(quantization_config, dict):
        quant_method = quantization_config["quant_method"]
    else:
        quant_method = getattr(quantization_config.quant_method, "value", quantization_config.quant_method)
    return str(quant_method)


@dataclass(frozen=True)
class GptqTextConfig:
    quantize_config: Any
    quant_linear_cls: type[nn.Module]


def resolve_gptq_text_config(quantization_config, device: str) -> GptqTextConfig | None:
    if quantization_config is None:
        return None
    assert quant_method_name(quantization_config) == "gptq", "qwen3 only supports GPTQ quantized weights"

    from gptqmodel.quantization import METHOD
    from gptqmodel.quantization.config import QuantizeConfig
    from gptqmodel.utils.importer import hf_select_quant_linear_v2

    raw_config = dict(quantization_config if isinstance(quantization_config, dict) else quantization_config.to_dict())
    quantize_config = QuantizeConfig.from_quant_config(raw_config)
    assert not quantize_config.lm_head, "qwen3 does not support quantized lm_head"
    quant_linear_cls = hf_select_quant_linear_v2(
        bits=quantize_config.bits,
        group_size=quantize_config.group_size,
        desc_act=quantize_config.desc_act,
        sym=quantize_config.sym,
        format=raw_config.get("checkpoint_format", raw_config.get("format", quantize_config.format)),
        quant_method=METHOD.GPTQ,
        device_map={"": device},
        backend=raw_config.get("backend") or "gptq_marlin",
        pack=False,
        dtype=getattr(quantize_config, "dtype", None),
        meta=quantize_config.meta,
    )
    return GptqTextConfig(quantize_config=quantize_config, quant_linear_cls=quant_linear_cls)


def build_text_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    text_quant_config: GptqTextConfig | None,
) -> nn.Module:
    return build_quant_linear(in_features, out_features, bias=bias, quant_config=text_quant_config)


def build_quant_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    quant_config: GptqTextConfig | None,
) -> nn.Module:
    if quant_config is None:
        return nn.Linear(in_features, out_features, bias=bias)

    quantize_config = quant_config.quantize_config
    layer = quant_config.quant_linear_cls(
        bits=quantize_config.bits,
        group_size=quantize_config.group_size,
        desc_act=quantize_config.desc_act,
        sym=quantize_config.sym,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        pack_dtype=quantize_config.pack_dtype,
        adapter=quantize_config.adapter,
        register_buffers=True,
        dev=torch.device("meta"),
    )
    layer.requires_grad_(False)
    return layer


def build_visual_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    visual_quant_config: GptqTextConfig | None,
) -> nn.Module:
    if visual_quant_config is None or out_features % 64 != 0:
        return nn.Linear(in_features, out_features, bias=bias)
    group_size = visual_quant_config.quantize_config.group_size
    if group_size != -1 and in_features % group_size != 0:
        return nn.Linear(in_features, out_features, bias=bias)
    return build_quant_linear(in_features, out_features, bias=bias, quant_config=visual_quant_config)


def make_visual_marlin_shapes(config) -> None:
    group_size = 128
    config.intermediate_size = ((config.intermediate_size + group_size - 1) // group_size) * group_size
