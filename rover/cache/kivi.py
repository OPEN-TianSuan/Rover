from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from rover.cache.base import BaseKVCache
from rover.kernels import loader as kernels
from rover.models.spec import ModelShape
from rover.nn.attention import repeat_kv
from rover.runtime.config import RuntimeConfig

if TYPE_CHECKING:
    from rover.runtime.types import Range
    from rover.storage.files import CacheFile


# ---------------------------------------------------------------------------
# Bit-packing utilities
# ---------------------------------------------------------------------------


def _unpack_along_last_dim(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack int32 tensor back to uint8 along the last dimension."""
    feat_per_int = 32 // bits
    mask = (1 << bits) - 1
    shifts = torch.arange(feat_per_int, device=packed.device) * bits
    unpacked = (packed.unsqueeze(-1) >> shifts) & mask
    *batch, n_packed, _ = unpacked.shape
    return unpacked.reshape(*batch, n_packed * feat_per_int).to(torch.uint8)


def _quantize_and_pack_along_last_dim(
    x: torch.Tensor, group_size: int, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize x asymmetrically along the last dimension and pack.

    Returns: (packed, scale, zero)
    """
    max_int = (1 << bits) - 1
    feat_per_int = 32 // bits
    *batch, N = x.shape
    assert N % group_size == 0
    num_groups = N // group_size

    grouped = x.reshape(*batch, num_groups, group_size)
    mn = grouped.amin(dim=-1)
    scale = ((grouped.amax(dim=-1) - mn) / max_int).clamp_min(1e-12)

    quantized = ((grouped - mn.unsqueeze(-1)) / scale.unsqueeze(-1)).round_().clamp_(0, max_int).to(torch.uint8)
    quantized = quantized.reshape(*batch, N)

    assert N % feat_per_int == 0
    t = quantized.reshape(*batch, N // feat_per_int, feat_per_int).to(torch.int32)
    shifts = torch.arange(feat_per_int, device=x.device) * bits
    packed = (t << shifts).sum(dim=-1).to(torch.int32)

    return packed, scale, mn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class KIVIConfig:
    bits: int
    group_size: int
    residual_length: int
    prefill_chunk_size: int
    device: torch.device
    kv_heads: int
    head_dim: int
    max_len: int


KIVI_DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# QuantizedBuffer: pre-allocated (packed, scale, zero) triple
# ---------------------------------------------------------------------------


class QuantizedBuffer:
    """Pre-allocated buffer of quantized data that grows along a specified dimension.

    Stores three tensors (packed, scale, zero) and tracks how many scale/zero slots
    are filled along `grow_dim`. `packed_per_scale` is the ratio of packed slots per
    scale/zero slot along the grow dimension.
    """

    def __init__(
        self, packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, grow_dim: int, packed_per_scale: int
    ):
        self._packed = packed
        self._scale = scale
        self._zero = zero
        self._grow_dim = grow_dim
        self._packed_per_scale = packed_per_scale
        self._len = 0  # filled scale/zero slots along grow_dim

    @property
    def packed(self) -> torch.Tensor:
        return self._packed.narrow(self._grow_dim, 0, self._len * self._packed_per_scale)

    @property
    def scale(self) -> torch.Tensor:
        return self._scale.narrow(self._grow_dim, 0, self._len)

    @property
    def zero(self) -> torch.Tensor:
        return self._zero.narrow(self._grow_dim, 0, self._len)

    def append(self, packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, count: int) -> None:
        d = self._grow_dim
        p_start = self._len * self._packed_per_scale
        self._packed.narrow(d, p_start, packed.shape[d]).copy_(packed)
        self._scale.narrow(d, self._len, count).copy_(scale)
        self._zero.narrow(d, self._len, count).copy_(zero)
        self._len += count

    def restore_length(self, length: int) -> None:
        self._len = length

    @property
    def max_nbytes(self) -> int:
        return self._packed.nbytes + self._scale.nbytes + self._zero.nbytes

    def write_to(self, file: CacheFile, offset: int) -> None:
        if self._len == 0:
            return
        file.write_tensor(self.packed.contiguous(), offset)
        file.write_tensor(self.scale.contiguous(), offset + self._packed.nbytes)
        file.write_tensor(self.zero.contiguous(), offset + self._packed.nbytes + self._scale.nbytes)

    def read_from(self, file: CacheFile, offset: int) -> None:
        if self._len == 0:
            return
        d = self._grow_dim
        n = self._len
        p_len = n * self._packed_per_scale

        packed_tmp = self._packed.narrow(d, 0, p_len).contiguous()
        file.read_tensor(packed_tmp, packed_tmp.nbytes, offset)
        self._packed.narrow(d, 0, p_len).copy_(packed_tmp)

        scale_off = offset + self._packed.nbytes
        scale_tmp = self._scale.narrow(d, 0, n).contiguous()
        file.read_tensor(scale_tmp, scale_tmp.nbytes, scale_off)
        self._scale.narrow(d, 0, n).copy_(scale_tmp)

        zero_off = scale_off + self._scale.nbytes
        zero_tmp = self._zero.narrow(d, 0, n).contiguous()
        file.read_tensor(zero_tmp, zero_tmp.nbytes, zero_off)
        self._zero.narrow(d, 0, n).copy_(zero_tmp)


# ---------------------------------------------------------------------------
# Quantized prefix: thin wrappers specifying shape and grow_dim
# ---------------------------------------------------------------------------


class QuantizedKeyPrefix(QuantizedBuffer):
    """Quantized key prefix. Keys quantized along token dim, stored transposed.

    Layout: (kv_heads, head_dim, tokens), grows along dim=2.
    Each scale/zero slot = 1 group. packed_per_scale = group_size // feat_per_int.
    """

    def __init__(self, *, kv_heads: int, head_dim: int, max_len: int, group_size: int, bits: int, device: torch.device):
        f = 32 // bits
        max_groups = max_len // group_size
        packed = torch.zeros((kv_heads, head_dim, max_groups * group_size // f), dtype=torch.int32, device=device)
        scale = torch.zeros((kv_heads, head_dim, max_groups), dtype=KIVI_DTYPE, device=device)
        zero = torch.zeros((kv_heads, head_dim, max_groups), dtype=KIVI_DTYPE, device=device)
        super().__init__(packed, scale, zero, grow_dim=2, packed_per_scale=group_size // f)
        self.group_size = group_size
        self.bits = bits

    @property
    def prefix_len(self) -> int:
        return self._len * self.group_size


class QuantizedValuePrefix(QuantizedBuffer):
    """Quantized value prefix. Values quantized along head_dim.

    Layout: (kv_heads, tokens, head_dim), grows along dim=1.
    Each scale/zero slot = 1 token. packed_per_scale = 1.
    """

    def __init__(self, *, kv_heads: int, head_dim: int, max_len: int, group_size: int, bits: int, device: torch.device):
        f = 32 // bits
        num_hd_groups = head_dim // group_size
        packed = torch.zeros((kv_heads, max_len, head_dim // f), dtype=torch.int32, device=device)
        scale = torch.zeros((kv_heads, max_len, num_hd_groups), dtype=KIVI_DTYPE, device=device)
        zero = torch.zeros((kv_heads, max_len, num_hd_groups), dtype=KIVI_DTYPE, device=device)
        super().__init__(packed, scale, zero, grow_dim=1, packed_per_scale=1)
        self.group_size = group_size
        self.bits = bits

    @property
    def prefix_len(self) -> int:
        return self._len


# ---------------------------------------------------------------------------
# KIVI cache implementation
# ---------------------------------------------------------------------------


class LayerKIVICache:
    def __init__(self, config: KIVIConfig):
        self.config = config
        self.key_prefix = QuantizedKeyPrefix(
            kv_heads=config.kv_heads,
            head_dim=config.head_dim,
            max_len=config.max_len,
            group_size=config.group_size,
            bits=config.bits,
            device=config.device,
        )
        self.value_prefix = QuantizedValuePrefix(
            kv_heads=config.kv_heads,
            head_dim=config.head_dim,
            max_len=config.max_len,
            group_size=config.group_size,
            bits=config.bits,
            device=config.device,
        )
        max_residual = config.residual_length + config.group_size - 1 + config.prefill_chunk_size
        self._key_residual = torch.zeros(
            (config.kv_heads, max_residual, config.head_dim), device=config.device, dtype=KIVI_DTYPE
        )
        self._value_residual = torch.zeros(
            (config.kv_heads, max_residual, config.head_dim), device=config.device, dtype=KIVI_DTYPE
        )
        self._residual_len = 0
        self.total_len = 0

    @property
    def prefix_len(self) -> int:
        return self.key_prefix.prefix_len

    @property
    def residual_len(self) -> int:
        return self._residual_len

    @property
    def key_residual(self) -> torch.Tensor:
        return self._key_residual[:, : self._residual_len, :]

    @property
    def value_residual(self) -> torch.Tensor:
        return self._value_residual[:, : self._residual_len, :]

    def update_range(self, start: int, end: int, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        assert start == self.total_len, f"only append supported: start={start} != total_len={self.total_len}"
        n = end - start
        pos = self._residual_len
        self._key_residual[:, pos : pos + n, :].copy_(key_states[0])
        self._value_residual[:, pos : pos + n, :].copy_(value_states[0])
        self._residual_len += n
        self.total_len = end
        self._flush_residual()

    def restore(self, prefix_len: int, residual_len: int) -> None:
        gs = self.config.group_size
        self.key_prefix.restore_length(prefix_len // gs)
        self.value_prefix.restore_length(prefix_len)
        self._residual_len = residual_len
        self.total_len = prefix_len + residual_len

    def _flush_residual(self) -> None:
        flushable = self._residual_len - self.config.residual_length
        flushable = max(flushable, 0)
        flushable = (flushable // self.config.group_size) * self.config.group_size
        if flushable == 0:
            return
        gs = self.config.group_size
        bits = self.config.bits
        for i in range(0, flushable, gs):
            keys_t = self._key_residual[:, i : i + gs, :].transpose(1, 2).contiguous()
            kp, ks, kz = _quantize_and_pack_along_last_dim(keys_t, gs, bits)
            self.key_prefix.append(kp, ks, kz, count=1)

            vp, vs, vz = _quantize_and_pack_along_last_dim(self._value_residual[:, i : i + gs, :], gs, bits)
            self.value_prefix.append(vp, vs, vz, count=gs)
        remaining = self._residual_len - flushable
        if remaining > 0:
            self._key_residual[:, :remaining, :].copy_(
                self._key_residual[:, flushable : flushable + remaining, :].clone()
            )
            self._value_residual[:, :remaining, :].copy_(
                self._value_residual[:, flushable : flushable + remaining, :].clone()
            )
        self._residual_len = remaining


class KIVIKVCache(BaseKVCache):
    """KIVI quantized KV cache with incremental disk IO.

    Disk layout per file (k or v), per layer:
        [prefix: packed | scale | zero] [residual: dense tokens]
    K and V files have different layouts (different prefix tensor shapes).
    """

    @staticmethod
    def compute_file_sizes(*, config: ModelShape, runtime_config: RuntimeConfig, max_len: int) -> tuple[int, int]:
        tc = config.text_config
        elem = torch.empty(0, dtype=KIVI_DTYPE).element_size()
        f = 32 // runtime_config.kivi_bits
        gs = runtime_config.kivi_group_size
        kv_heads, head_dim = tc.num_key_value_heads, tc.head_dim
        max_groups = max_len // gs
        key_prefix_bytes = (
            kv_heads * head_dim * (max_groups * gs // f) * 4 + kv_heads * head_dim * max_groups * elem * 2
        )
        value_prefix_bytes = kv_heads * max_len * (head_dim // f) * 4 + kv_heads * max_len * (head_dim // gs) * elem * 2
        residual_bytes = kv_heads * (runtime_config.kivi_residual_length + gs - 1) * head_dim * elem
        return (
            (key_prefix_bytes + residual_bytes) * tc.num_hidden_layers,
            (value_prefix_bytes + residual_bytes) * tc.num_hidden_layers,
        )

    @staticmethod
    def build(
        *,
        config: ModelShape,
        runtime_config: RuntimeConfig,
        max_len: int,
        cache_metadata: dict | None = None,
    ) -> KIVIKVCache:
        tc = config.text_config
        return KIVIKVCache(
            num_layers=tc.num_hidden_layers,
            cache_metadata=cache_metadata,
            config=KIVIConfig(
                bits=runtime_config.kivi_bits,
                group_size=runtime_config.kivi_group_size,
                residual_length=runtime_config.kivi_residual_length,
                prefill_chunk_size=runtime_config.prefill_chunk_size,
                device=torch.device(runtime_config.device),
                kv_heads=tc.num_key_value_heads,
                head_dim=tc.head_dim,
                max_len=max_len,
            ),
        )

    def __init__(self, *, num_layers: int, config: KIVIConfig, cache_metadata: dict | None = None):
        self.config = config
        self.layers = [LayerKIVICache(config) for _ in range(num_layers)]
        self._num_layers = num_layers
        elem = torch.empty(0, dtype=KIVI_DTYPE).element_size()
        self._residual_file_bytes = (
            config.kv_heads * (config.residual_length + config.group_size - 1) * config.head_dim * elem
        )
        self._decode_base_len = cache_metadata["decode_base_len"] if cache_metadata else None
        self._decode_prefill_lengths = cache_metadata["decode_prefill_lengths"] if cache_metadata else None
        if cache_metadata:
            for layer, (prefix_len, residual_len) in enumerate(cache_metadata["kv_lengths"]):
                self.layers[layer].restore(prefix_len, residual_len)

    def cache_format(self) -> str:
        return "kivi"

    def update_kv(self, r: Range, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.layers[r.layer].update_range(r.start, r.end, key_states, value_states)

    def get_kv_for_attention(self, layer: int) -> LayerKIVICache:
        return self.layers[layer]

    def get_metadata(self) -> dict:
        kv_lengths = self._decode_prefill_lengths
        if kv_lengths is None:
            kv_lengths = [(lc.prefix_len, lc.residual_len) for lc in self.layers]
        return {
            "kv_lengths": kv_lengths,
            "decode_base_len": self._decode_base_len,
            "decode_prefill_lengths": self._decode_prefill_lengths,
        }

    def prefill_key_file_size(self) -> int:
        lc = self.layers[0]
        return (lc.key_prefix.max_nbytes + self._residual_file_bytes) * self._num_layers

    def prefill_value_file_size(self) -> int:
        lc = self.layers[0]
        return (lc.value_prefix.max_nbytes + self._residual_file_bytes) * self._num_layers

    def save_prefill_layer(self, layer: int, start: int, end: int, k_file: CacheFile, v_file: CacheFile) -> None:
        lc = self.layers[layer]
        k_layer_bytes = lc.key_prefix.max_nbytes + self._residual_file_bytes
        v_layer_bytes = lc.value_prefix.max_nbytes + self._residual_file_bytes
        k_offset = layer * k_layer_bytes
        v_offset = layer * v_layer_bytes
        lc.key_prefix.write_to(k_file, k_offset)
        lc.value_prefix.write_to(v_file, v_offset)
        if lc.residual_len > 0:
            k_file.write_tensor(lc.key_residual.contiguous(), k_offset + lc.key_prefix.max_nbytes)
            v_file.write_tensor(lc.value_residual.contiguous(), v_offset + lc.value_prefix.max_nbytes)

    def load_prefill_layer(self, layer: int, start: int, end: int, k_file: CacheFile, v_file: CacheFile) -> None:
        lc = self.layers[layer]
        k_layer_bytes = lc.key_prefix.max_nbytes + self._residual_file_bytes
        v_layer_bytes = lc.value_prefix.max_nbytes + self._residual_file_bytes
        k_offset = layer * k_layer_bytes
        v_offset = layer * v_layer_bytes
        lc.key_prefix.read_from(k_file, k_offset)
        lc.value_prefix.read_from(v_file, v_offset)
        if lc.residual_len > 0:
            tmp_k = lc._key_residual[:, : lc.residual_len, :].contiguous()
            k_file.read_tensor(tmp_k, tmp_k.nbytes, k_offset + lc.key_prefix.max_nbytes)
            lc._key_residual[:, : lc.residual_len, :].copy_(tmp_k)

            tmp_v = lc._value_residual[:, : lc.residual_len, :].contiguous()
            v_file.read_tensor(tmp_v, tmp_v.nbytes, v_offset + lc.value_prefix.max_nbytes)
            lc._value_residual[:, : lc.residual_len, :].copy_(tmp_v)

    def begin_decode(self, prefix_len: int) -> None:
        self._decode_base_len = prefix_len
        self._decode_prefill_lengths = [(lc.prefix_len, lc.residual_len) for lc in self.layers]

    def decode_record_nbytes(self) -> int:
        elem = torch.empty(0, dtype=KIVI_DTYPE).element_size()
        token_nbytes = self.config.kv_heads * self.config.head_dim * elem
        return self._num_layers * 2 * token_nbytes

    def save_decode_tokens(self, start_token: int, end_token: int, decode_file: CacheFile) -> None:
        assert self._decode_base_len is not None
        assert end_token >= start_token
        record_nbytes = self.decode_record_nbytes()
        for token in range(start_token, end_token):
            absolute_token = self._decode_base_len + token
            buffers = []
            for layer_cache in self.layers:
                residual_start = layer_cache.prefix_len
                residual_end = residual_start + layer_cache.residual_len
                assert residual_start <= absolute_token < residual_end
                residual_token = absolute_token - residual_start
                k = layer_cache.key_residual[:, residual_token : residual_token + 1, :]
                v = layer_cache.value_residual[:, residual_token : residual_token + 1, :]
                buffers.append(k.contiguous().cpu().view(torch.uint8).numpy().tobytes())
                buffers.append(v.contiguous().cpu().view(torch.uint8).numpy().tobytes())
            decode_file.writev(buffers, token * record_nbytes)

    def load_decode_tokens(self, decode_file: CacheFile, num_tokens: int) -> None:
        from rover.runtime.types import Range

        assert self._decode_base_len is not None
        token_nbytes = self.config.kv_heads * self.config.head_dim * torch.empty(0, dtype=KIVI_DTYPE).element_size()
        record_nbytes = self.decode_record_nbytes()
        for token in range(num_tokens):
            absolute_token = self._decode_base_len + token
            raw = bytearray(decode_file.read(record_nbytes, token * record_nbytes))
            offset = 0
            for layer, _ in enumerate(self.layers):
                k = torch.frombuffer(
                    raw, dtype=KIVI_DTYPE, count=self.config.kv_heads * self.config.head_dim, offset=offset
                )
                offset += token_nbytes
                v = torch.frombuffer(
                    raw, dtype=KIVI_DTYPE, count=self.config.kv_heads * self.config.head_dim, offset=offset
                )
                offset += token_nbytes
                self.update_kv(
                    Range(layer, absolute_token, absolute_token + 1),
                    k.reshape(1, self.config.kv_heads, 1, self.config.head_dim).to(self.config.device),
                    v.reshape(1, self.config.kv_heads, 1, self.config.head_dim).to(self.config.device),
                )


def bmm_fA_qB_outer(
    fA: torch.Tensor,
    qB: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bits: int,
    group_size: int,
    num_heads: int,
    kv_heads: int,
) -> torch.Tensor:
    """Batched quantized matmul: output = fA @ dequant(qB)."""
    batch, num_heads_arg, M, IC = fA.shape
    N = qB.shape[-1] * (32 // bits)

    fA_flat = fA.reshape(batch * num_heads_arg, M, IC).contiguous()
    qB_t = (
        qB.unsqueeze(0)
        .expand(batch, -1, -1, -1)
        .reshape(batch * kv_heads, IC, qB.shape[-1])
        .transpose(1, 2)
        .contiguous()
    )
    sc_t = (
        scales.unsqueeze(0)
        .expand(batch, -1, -1, -1)
        .reshape(batch * kv_heads, IC, scales.shape[-1])
        .transpose(1, 2)
        .contiguous()
    )
    zr_t = (
        zeros.unsqueeze(0)
        .expand(batch, -1, -1, -1)
        .reshape(batch * kv_heads, IC, zeros.shape[-1])
        .transpose(1, 2)
        .contiguous()
    )

    out = kernels.gemv_forward_cuda_outer_dim(fA_flat, qB_t, sc_t, zr_t, bits, group_size, num_heads_arg, kv_heads)
    return out.view(batch, num_heads_arg, M, N)


def kivi_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    layer_cache: LayerKIVICache,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch, num_heads, q_len, head_dim = query.shape
    prefix_len = layer_cache.prefix_len
    residual_len = layer_cache.residual_len
    kv_heads = layer_cache.config.kv_heads

    attn_scores_list = []

    if prefix_len > 0:
        kp = layer_cache.key_prefix
        scores_prefix = (
            bmm_fA_qB_outer(
                query,
                kp.packed.to(query.device),
                kp.scale.to(query.device),
                kp.zero.to(query.device),
                kp.bits,
                kp.group_size,
                num_heads,
                kv_heads,
            )
            * scaling
        )
        attn_scores_list.append(scores_prefix)

    if residual_len > 0:
        key_residual = layer_cache.key_residual
        key_residual = repeat_kv(key_residual.unsqueeze(0), num_key_value_groups)
        attn_scores_list.append(torch.matmul(query, key_residual.transpose(-2, -1)) * scaling)

    attn_scores = torch.cat(attn_scores_list, dim=-1)
    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask
    attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)

    attn_output_list = []

    if prefix_len > 0:
        vp = layer_cache.value_prefix
        attn_w_prefix = attn_weights[:, :, :, :prefix_len]
        out_prefix = bmm_fA_qB_outer(
            attn_w_prefix,
            vp.packed.to(query.device),
            vp.scale.to(query.device),
            vp.zero.to(query.device),
            vp.bits,
            vp.group_size,
            num_heads,
            kv_heads,
        )
        attn_output_list.append(out_prefix)

    if residual_len > 0:
        value_residual = layer_cache.value_residual
        value_residual = repeat_kv(value_residual.unsqueeze(0), num_key_value_groups)
        attn_w_residual = attn_weights[:, :, :, prefix_len:]
        attn_output_list.append(torch.matmul(attn_w_residual, value_residual))

    attn_output = sum(attn_output_list) if len(attn_output_list) > 1 else attn_output_list[0]
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch, q_len, num_heads * head_dim)
    return attn_output, None
