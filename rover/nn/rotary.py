from dataclasses import dataclass

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@dataclass(frozen=True)
class RotaryEmbedding:
    _cos: torch.Tensor
    _sin: torch.Tensor

    @classmethod
    def from_freqs(cls, freqs: torch.Tensor) -> "RotaryEmbedding":
        emb = torch.cat((freqs, freqs), dim=-1)
        return cls(emb.cos().float(), emb.sin().float())

    def slice(self, start: int, end: int) -> "RotaryEmbedding":
        return RotaryEmbedding(self._cos[:, start:end, :], self._sin[:, start:end, :])

    def apply(self, q, k):
        orig_dtype = q.dtype
        q, k = q.float(), k.float()
        cos = self._cos.unsqueeze(1)
        sin = self._sin.unsqueeze(1)
        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)
        return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


__all__ = ["RotaryEmbedding", "apply_rotary_pos_emb"]
