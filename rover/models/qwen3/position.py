"""Qwen3 spatial and temporal rotary positions."""

import torch


def build_rotary_inv_freq(text_config) -> torch.Tensor:
    head_dim = text_config.head_dim or text_config.hidden_size // text_config.num_attention_heads
    rope_params = text_config.rope_parameters
    assert rope_params is not None and "rope_theta" in rope_params, "rope_theta required in rope_parameters"
    base = rope_params["rope_theta"]
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))


def build_vision_rotary_inv_freq(vision_config) -> torch.Tensor:
    head_dim = vision_config.hidden_size // vision_config.num_heads
    dim = head_dim // 2
    return 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
    """Reorganize frequency layout from chunked [TTT...HHH...WWW] to interleaved [THWTHW...].

    Args:
        freqs: (3, batch, seq_len, head_dim // 2)
        mrope_section: [T_size, H_size, W_size] summing to head_dim // 2
    Returns:
        (batch, seq_len, head_dim // 2)
    """
    freqs_out = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_out[..., idx] = freqs[dim, ..., idx]
    return freqs_out
