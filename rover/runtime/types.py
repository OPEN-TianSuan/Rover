"""Shared request progress and prefill input types."""

from dataclasses import dataclass
from enum import Enum

import torch

from rover.nn.rotary import RotaryEmbedding


@dataclass(frozen=True)
class Range:
    layer: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ChunkLayerIndex:
    chunk_id: int
    layer: int
    start: int
    end: int


@dataclass(frozen=True)
class PrefillInputs:
    inputs_embeds: torch.Tensor
    pos_emb: "RotaryEmbedding"
    visual_pos_masks: torch.Tensor
    deepstack_visual_embeds: list[torch.Tensor]
    decode_rope_offset: int


class Phase(Enum):
    PREPARE_PREFILL = "prepare_prefill"
    PREFILLING = "prefilling"
    PREPARE_DECODING = "prepare_decoding"
    DECODING_DENSE_OVERLAP = "decoding_dense_overlap"
    DECODING = "decoding"
