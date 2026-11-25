from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from rover.cache.base import BaseKVCache
from rover.diagnostics.tracing import tracer
from rover.models.spec import ModelShape
from rover.runtime.config import RuntimeConfig

if TYPE_CHECKING:
    from rover.runtime.state import GenerationState
    from rover.runtime.types import Range
    from rover.storage.files import CacheFile


class DenseKVCache(BaseKVCache):
    """Dense KV cache. Layout: (num_layers, 1, kv_heads, max_len, head_dim)."""

    METADATA_FILE = "dense_lengths.json"

    @staticmethod
    def compute_file_sizes(*, config: ModelShape, runtime_config: RuntimeConfig, max_len: int) -> tuple[int, int]:
        tc = config.text_config
        elem = torch.empty(0, dtype=torch.bfloat16).element_size()
        size = tc.num_hidden_layers * max_len * tc.num_key_value_heads * tc.head_dim * elem
        return size, size

    @staticmethod
    def build(
        *,
        config: ModelShape,
        runtime_config: RuntimeConfig,
        max_len: int,
        cache_metadata: dict | None = None,
    ) -> DenseKVCache:
        tc = config.text_config
        return DenseKVCache(
            num_layers=tc.num_hidden_layers,
            kv_heads=tc.num_key_value_heads,
            max_len=max_len,
            head_dim=tc.head_dim,
            device=runtime_config.device,
            cache_metadata=cache_metadata,
        )

    def __init__(
        self,
        *,
        num_layers: int,
        kv_heads: int,
        max_len: int,
        head_dim: int,
        device: str,
        cache_metadata: dict | None = None,
    ):
        with torch.device(device):
            self._k = torch.zeros((num_layers, 1, kv_heads, max_len, head_dim), dtype=torch.bfloat16)
            self._v = torch.zeros((num_layers, 1, kv_heads, max_len, head_dim), dtype=torch.bfloat16)
        self._num_layers = num_layers
        self._kv_heads = kv_heads
        self._max_len = max_len
        self._head_dim = head_dim
        self._saved_len = [0] * num_layers
        self._decode_base_len = cache_metadata["decode_base_len"] if cache_metadata else None
        if cache_metadata and "cur_len" in cache_metadata:
            self._cur_len = list(cache_metadata["cur_len"])
        elif self._decode_base_len is not None:
            self._cur_len = [self._decode_base_len] * num_layers
        else:
            self._cur_len = [0] * num_layers

    def cache_format(self) -> str:
        return "dense"

    def update_kv(self, r: Range, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self._k[r.layer, :, :, r.start : r.end, :].copy_(key_states)
        self._v[r.layer, :, :, r.start : r.end, :].copy_(value_states)
        self._cur_len[r.layer] = max(self._cur_len[r.layer], r.end)

    def get_kv(self, r: Range) -> tuple[torch.Tensor, torch.Tensor]:
        return self._k[r.layer, :, :, r.start : r.end, :], self._v[r.layer, :, :, r.start : r.end, :]

    def current_length(self, layer: int) -> int:
        return self._cur_len[layer]

    def get_prefix(self, layer: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._k[layer, :, :, :length, :], self._v[layer, :, :, :length, :]

    def get_kv_for_attention(self, layer: int) -> None:
        return None

    def _token_stride(self) -> int:
        return self._kv_heads * self._head_dim * self._k.element_size()

    def _offset(self, layer: int, token: int) -> int:
        return (layer * self._max_len + token) * self._token_stride()

    def prefill_key_file_size(self) -> int:
        return self._num_layers * self._max_len * self._token_stride()

    def prefill_value_file_size(self) -> int:
        return self._num_layers * self._max_len * self._token_stride()

    def get_metadata(self) -> dict:
        prefix_len = self._decode_base_len
        if prefix_len is None:
            cur_len = list(self._cur_len)
        else:
            cur_len = [min(length, prefix_len) for length in self._cur_len]
        return {"cur_len": cur_len, "decode_base_len": self._decode_base_len}

    def metadata_payload(self, base_dir: Path) -> dict:
        if self._decode_base_len is not None:
            return {"decode_base_len": self._decode_base_len}
        sidecar = base_dir / self.METADATA_FILE
        with open(sidecar, "w") as f:
            json.dump({"cur_len": self._cur_len}, f)
        return {"decode_base_len": None, "lengths_file": self.METADATA_FILE}

    @classmethod
    def resolve_metadata_payload(cls, base_dir: Path, payload: dict) -> dict:
        if "lengths_file" not in payload:
            return payload
        with open(base_dir / payload["lengths_file"]) as f:
            return {**payload, **json.load(f)}

    def save_prefill_layer(self, layer: int, start: int, end: int, k_file: CacheFile, v_file: CacheFile) -> None:
        if end <= start:
            return
        offset = self._offset(layer, start)
        k_file.write_tensor(self._k[layer, 0, :, start:end, :].permute(1, 0, 2), offset)
        v_file.write_tensor(self._v[layer, 0, :, start:end, :].permute(1, 0, 2), offset)
        self._saved_len[layer] = max(self._saved_len[layer], end)

    def load_prefill_layer(self, layer: int, start: int, end: int, k_file: CacheFile, v_file: CacheFile) -> None:
        if end <= start:
            return
        offset = self._offset(layer, start)
        nbytes = (end - start) * self._token_stride()
        k_file.read_tensor(self._k[layer, 0, :, start:end, :].permute(1, 0, 2), nbytes, offset)
        v_file.read_tensor(self._v[layer, 0, :, start:end, :].permute(1, 0, 2), nbytes, offset)
        self._saved_len[layer] = max(self._saved_len[layer], end)

    def begin_decode(self, prefix_len: int) -> None:
        self._decode_base_len = prefix_len

    def decode_record_nbytes(self) -> int:
        return self._num_layers * 2 * self._token_stride()

    def save_decode_tokens(self, start_token: int, end_token: int, decode_file: CacheFile) -> None:
        assert self._decode_base_len is not None
        assert end_token >= start_token
        record_nbytes = self.decode_record_nbytes()
        sync_cuda = self._k.device.type == "cuda"
        for token in range(start_token, end_token):
            absolute_token = self._decode_base_len + token
            buffers = []
            with tracer.timer("commit_save_kv_pack", cache_format="dense", sync_cuda=sync_cuda):
                for layer in range(self._num_layers):
                    k = self._k[layer, 0, :, absolute_token : absolute_token + 1, :].permute(1, 0, 2)
                    v = self._v[layer, 0, :, absolute_token : absolute_token + 1, :].permute(1, 0, 2)
                    buffers.append(k.contiguous().cpu().view(torch.uint8).numpy().tobytes())
                    buffers.append(v.contiguous().cpu().view(torch.uint8).numpy().tobytes())
            with tracer.timer("commit_save_kv_write", cache_format="dense", sync_cuda=sync_cuda):
                decode_file.writev(buffers, token * record_nbytes)

    def load_decode_tokens(self, decode_file: CacheFile, num_tokens: int) -> None:
        assert self._decode_base_len is not None
        token_stride = self._token_stride()
        record_nbytes = self.decode_record_nbytes()
        for token in range(num_tokens):
            absolute_token = self._decode_base_len + token
            raw = bytearray(decode_file.read(record_nbytes, token * record_nbytes))
            offset = 0
            for layer in range(self._num_layers):
                k = torch.frombuffer(raw, dtype=self._k.dtype, count=self._kv_heads * self._head_dim, offset=offset)
                self._k[layer, 0, :, absolute_token, :].copy_(k.reshape(self._kv_heads, self._head_dim))
                offset += token_stride
                v = torch.frombuffer(raw, dtype=self._v.dtype, count=self._kv_heads * self._head_dim, offset=offset)
                self._v[layer, 0, :, absolute_token, :].copy_(v.reshape(self._kv_heads, self._head_dim))
                offset += token_stride
                self._cur_len[layer] = max(self._cur_len[layer], absolute_token + 1)
                self._saved_len[layer] = max(self._saved_len[layer], absolute_token + 1)


def text_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    return module._eager_attention_forward(query, key, value, attention_mask, scaling)


def dense_attention_forward(
    module: nn.Module,
    generation_state: GenerationState,
    r: Range,
    query_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    past_kv_range = type(r)(r.layer, 0, r.end)
    key_states, value_states = generation_state.kv_cache.get_kv(past_kv_range)
    sync_cuda = module.runtime_config.sync_cuda_timers and query_states.device.type == "cuda"
    with tracer.timer("decode_attn_total", cache_format="dense", sync_cuda=sync_cuda):
        with tracer.timer("decode_attn_sdpa", cache_format="dense", sync_cuda=sync_cuda):
            # SDPA handles GQA directly, which avoids materializing repeated K/V and full attention weights.
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states.to(query_states.dtype),
                value_states.to(query_states.dtype),
                attn_mask=module._build_attention_bias(r),
                dropout_p=0.0,
                scale=module.scaling,
                enable_gqa=module.num_key_value_groups > 1,
            )
        with tracer.timer("decode_attn_output", cache_format="dense", sync_cuda=sync_cuda):
            attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None
