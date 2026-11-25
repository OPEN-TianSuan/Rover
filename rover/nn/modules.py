"""Small PyTorch building blocks used by the inference-only model."""

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

ACT2FN = {"silu": F.silu, "gelu": F.gelu, "relu": F.relu, "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh")}


class InferenceModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device


@dataclass
class VisionOutput:
    last_hidden_state: torch.Tensor | None = None
    pooler_output: torch.Tensor | None = None
    hidden_states: tuple | None = None
    attentions: tuple | None = None


def maybe_autocast(device_type, enabled=True):
    if enabled or torch.is_autocast_enabled(device_type):
        return torch.autocast(device_type, enabled=enabled)
    return nullcontext()
