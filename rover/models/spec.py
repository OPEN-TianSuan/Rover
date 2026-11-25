"""Structural model dimensions shared by cache implementations."""

from typing import Protocol


class TextShape(Protocol):
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int


class ModelShape(Protocol):
    text_config: TextShape
