"""Captured decode graph handles."""

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class DecodeGraphRunner:
    logits_out: torch.Tensor
    replay_step: Callable[[int, int, int], None]
