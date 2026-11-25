from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import torch

from rover.cache.base import BaseKVCache
from rover.cache.constants import LLOYD_MAX_BOUNDARIES, LLOYD_MAX_CENTROIDS
from rover.diagnostics.tracing import tracer
from rover.kernels import loader as _ext
from rover.models.spec import ModelShape
from rover.runtime.config import RuntimeConfig

if TYPE_CHECKING:
    from rover.cache.dense import DenseKVCache
    from rover.runtime.state import GenerationState
    from rover.runtime.types import Range
    from rover.storage.files import CacheFile


TURBOQUANT_DTYPE = torch.bfloat16
PROD_QJL_SHRINK = 0.25


def _feat_per_int(bits: int) -> int:
    return 32 // bits


def _packed_dim(dim: int, bits: int) -> int:
    return (dim + _feat_per_int(bits) - 1) // _feat_per_int(bits)


def _unpack_codes(packed: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    if packed.dtype != torch.int32:
        packed = packed.to(torch.int32)
    feat = _feat_per_int(bits)
    mask = (1 << bits) - 1
    shifts = torch.arange(feat, device=packed.device, dtype=torch.int32) * bits
    unpacked = ((packed.unsqueeze(-1) >> shifts) & mask).reshape(*packed.shape[:-1], -1)
    return unpacked[..., :dim].to(torch.int32)


def _random_orthogonal(dim: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    mat = torch.randn((dim, dim), generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(mat)
    q = q * torch.sign(torch.diag(r)).unsqueeze(0)
    return q.to(device=device)


def _random_gaussian(dim: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn((dim, dim), generator=generator, dtype=torch.float32).to(device=device)


class TurboQuantMSEQuantizer:
    def __init__(self, dim: int, bits: int, seed: int, device: torch.device):
        assert 1 <= bits <= 4
        self.dim = dim
        self.bits = bits
        self.rotation = _random_orthogonal(dim, seed, device).contiguous()
        self.rotation_t = self.rotation.transpose(0, 1).contiguous()
        sigma = 1.0 / math.sqrt(dim)
        self.centroids = torch.tensor(LLOYD_MAX_CENTROIDS[bits], device=device, dtype=torch.float32) * sigma
        self.boundaries = torch.tensor(LLOYD_MAX_BOUNDARIES[bits], device=device, dtype=torch.float32) * sigma

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.to(torch.float32)
        norms = torch.linalg.vector_norm(x, dim=-1)
        y = (x / norms.unsqueeze(-1).clamp_min(1e-10)) @ self.rotation_t
        return torch.bucketize(y, self.boundaries).to(torch.int32), norms

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        return (self.centroids[indices.to(torch.long)] @ self.rotation) * norms.to(torch.float32).unsqueeze(-1)


class TurboQuantProdQuantizer:
    def __init__(self, dim: int, bits: int, seed: int, device: torch.device):
        assert bits >= 2
        self.dim = dim
        self.bits = bits
        self.mse_quantizer = TurboQuantMSEQuantizer(dim, bits - 1, seed, device)
        self.S = _random_gaussian(dim, seed + 1, device)
        self.S_t = self.S.transpose(0, 1).contiguous()
        self.coeff = PROD_QJL_SHRINK * math.sqrt(math.pi / 2.0) / dim

    def dequantize(
        self,
        mse_indices: torch.Tensor,
        norms: torch.Tensor,
        qjl: torch.Tensor,
        residual_norms: torch.Tensor,
    ) -> torch.Tensor:
        x_mse = self.mse_quantizer.dequantize(mse_indices, norms)
        qjl_signed = qjl.to(torch.float32).mul(2.0).sub(1.0)
        x_qjl = self.coeff * residual_norms.to(torch.float32).unsqueeze(-1) * (qjl_signed @ self.S)
        return x_mse + x_qjl


class MSETokenStore:
    def __init__(
        self,
        *,
        quantizer: TurboQuantMSEQuantizer,
        kv_heads: int,
        max_len: int,
        device: torch.device,
    ):
        packed_dim = _packed_dim(quantizer.dim, quantizer.bits)
        self.quantizer = quantizer
        self._packed = torch.zeros((kv_heads, max_len, packed_dim), dtype=torch.int32, device=device)
        self._norms = torch.zeros((kv_heads, max_len, 1), dtype=torch.bfloat16, device=device)
        self._norms_f32 = torch.zeros((kv_heads, max_len, 1), dtype=torch.float32, device=device)

    @property
    def nbytes(self) -> int:
        return self._packed.nbytes + self._norms.nbytes

    @property
    def token_nbytes(self) -> int:
        return _token_slice_nbytes(self._packed) + _token_slice_nbytes(self._norms)

    def write(self, start: int, x: torch.Tensor) -> None:
        _ext.append_cache_mse(
            x.contiguous(),
            self.quantizer.rotation_t,
            self.quantizer.boundaries.contiguous(),
            self._packed,
            self._norms,
            self._norms_f32,
            int(start),
            int(self.quantizer.bits),
        )

    def dequantize(self, length: int, include_residual: bool = True) -> torch.Tensor:
        indices = _unpack_codes(self._packed[:, :length, :], self.quantizer.bits, self.quantizer.dim)
        return self.quantizer.dequantize(indices, self._norms_f32[:, :length, 0]).to(TURBOQUANT_DTYPE)

    def qk(
        self,
        query: torch.Tensor,
        total_len: int,
        num_heads: int,
        kv_heads: int,
        profiler: TurboQuantProfiler,
    ) -> torch.Tensor:
        batch, _, q_len, _ = query.shape
        assert q_len == 1
        query_f32 = query if query.dtype == torch.float32 else query.to(torch.float32)
        with profiler.scope("quant_qk_mse", query.device):
            scores = _ext.qk_mse(
                query_f32.reshape(batch * num_heads, q_len, self.quantizer.dim).contiguous(),
                self._packed,
                self._norms_f32[:, :, 0].contiguous(),
                self.quantizer.rotation_t,
                self.quantizer.centroids.contiguous(),
                int(total_len),
                int(self.quantizer.bits),
                int(num_heads),
                int(kv_heads),
            )
        return scores.view(batch, num_heads, q_len, total_len)

    def av(
        self,
        attn_weights: torch.Tensor,
        total_len: int,
        num_heads: int,
        kv_heads: int,
        profiler: TurboQuantProfiler,
    ) -> torch.Tensor:
        batch, _, q_len, _ = attn_weights.shape
        assert q_len == 1
        attn_f32 = attn_weights if attn_weights.dtype == torch.float32 else attn_weights.to(torch.float32)
        with profiler.scope("quant_av_mse_rotated", attn_weights.device):
            rotated = _ext.av_mse_rotated(
                attn_f32.reshape(batch * num_heads, q_len, total_len).contiguous(),
                self._packed,
                self._norms_f32[:, :, 0].contiguous(),
                self.quantizer.centroids.contiguous(),
                int(total_len),
                int(self.quantizer.bits),
                int(num_heads),
                int(kv_heads),
            )
        with profiler.scope("quant_av_mse_rotate_back", attn_weights.device):
            output = torch.matmul(rotated[..., : self.quantizer.dim], self.quantizer.rotation)
        return output.view(batch, num_heads, q_len, self.quantizer.dim).to(attn_weights.dtype)

    def write_range_to(self, file: CacheFile, offset: int, start: int, end: int) -> None:
        _write_token_range(file, self._packed, offset, start, end)
        _write_token_range(file, self._norms, offset + self._packed.nbytes, start, end)

    def read_from(self, file: CacheFile, offset: int) -> None:
        _read_token_range(file, self._packed, offset)
        norms_offset = offset + self._packed.nbytes
        _read_token_range(file, self._norms, norms_offset)
        self._norms_f32.copy_(self._norms.to(torch.float32))

    def append_token_buffers(self, buffers: list[bytes], token: int) -> None:
        _append_tensor_token_bytes(buffers, self._packed, token)
        _append_tensor_token_bytes(buffers, self._norms, token)

    def pack_token_record(self, record: torch.Tensor, token: int, offset: int) -> int:
        _ext.pack_token_slice(self._packed, record, int(token), int(offset))
        offset += _token_slice_nbytes(self._packed)
        _ext.pack_token_slice(self._norms, record, int(token), int(offset))
        offset += _token_slice_nbytes(self._norms)
        return offset

    def load_tokens(self, raw: bytearray, start_token: int, num_tokens: int, record_nbytes: int, offset: int) -> int:
        end_token = start_token + num_tokens
        offset = _load_tensor_tokens(raw, self._packed, start_token, num_tokens, record_nbytes, offset)
        offset = _load_tensor_tokens(raw, self._norms, start_token, num_tokens, record_nbytes, offset)
        self._norms_f32[:, start_token:end_token, :].copy_(self._norms[:, start_token:end_token, :].to(torch.float32))
        return offset


class ProdTokenStore:
    def __init__(
        self,
        *,
        quantizer: TurboQuantProdQuantizer,
        kv_heads: int,
        max_len: int,
        device: torch.device,
    ):
        self.quantizer = quantizer
        packed_mse_dim = _packed_dim(quantizer.dim, quantizer.mse_quantizer.bits)
        self._packed_mse = torch.zeros((kv_heads, max_len, packed_mse_dim), dtype=torch.int32, device=device)
        self._qjl = torch.zeros((kv_heads, max_len, _packed_dim(quantizer.dim, 1)), dtype=torch.int32, device=device)
        self._residual_norms = torch.zeros((kv_heads, max_len, 1), dtype=TURBOQUANT_DTYPE, device=device)
        self._residual_norms_f32 = torch.zeros((kv_heads, max_len, 1), dtype=torch.float32, device=device)
        self._norms = torch.zeros((kv_heads, max_len, 1), dtype=TURBOQUANT_DTYPE, device=device)
        self._norms_f32 = torch.zeros((kv_heads, max_len, 1), dtype=torch.float32, device=device)
        self._qjl_centroids = torch.tensor([-1.0, 1.0], dtype=torch.float32, device=device)

    @property
    def nbytes(self) -> int:
        return self._packed_mse.nbytes + self._qjl.nbytes + self._residual_norms.nbytes + self._norms.nbytes

    @property
    def token_nbytes(self) -> int:
        return (
            _token_slice_nbytes(self._packed_mse)
            + _token_slice_nbytes(self._qjl)
            + _token_slice_nbytes(self._residual_norms)
            + _token_slice_nbytes(self._norms)
        )

    def write(self, start: int, x: torch.Tensor) -> None:
        _ext.append_cache_prod(
            x.contiguous(),
            self.quantizer.mse_quantizer.rotation_t,
            self.quantizer.mse_quantizer.boundaries.contiguous(),
            self.quantizer.mse_quantizer.centroids.contiguous(),
            self.quantizer.S_t,
            self._packed_mse,
            self._qjl,
            self._residual_norms,
            self._residual_norms_f32,
            self._norms,
            self._norms_f32,
            int(start),
            int(self.quantizer.mse_quantizer.bits),
        )

    def dequantize(self, length: int, include_residual: bool = True) -> torch.Tensor:
        mse_indices = _unpack_codes(
            self._packed_mse[:, :length, :], self.quantizer.mse_quantizer.bits, self.quantizer.dim
        )
        norms = self._norms_f32[:, :length, 0]
        mse = self.quantizer.mse_quantizer.dequantize(mse_indices, norms)
        if not include_residual:
            return mse.to(TURBOQUANT_DTYPE)
        qjl = _unpack_codes(self._qjl[:, :length, :], 1, self.quantizer.dim)
        return self.quantizer.dequantize(
            mse_indices,
            norms,
            qjl,
            self._residual_norms_f32[:, :length, 0],
        ).to(TURBOQUANT_DTYPE)

    def qk(
        self,
        query: torch.Tensor,
        total_len: int,
        num_heads: int,
        kv_heads: int,
        profiler: TurboQuantProfiler,
    ) -> torch.Tensor:
        batch, _, q_len, dim = query.shape
        assert batch == 1 and q_len == 1
        with profiler.scope("quant_qk_prod", query.device):
            scores = _ext.qk_prod(
                query.reshape(batch * num_heads, q_len, dim).contiguous(),
                self._packed_mse,
                self._qjl,
                self._residual_norms_f32[:, :, 0].contiguous(),
                self._norms_f32[:, :, 0].contiguous(),
                self.quantizer.mse_quantizer.rotation_t,
                self.quantizer.S_t,
                self.quantizer.mse_quantizer.centroids.contiguous(),
                float(self.quantizer.coeff),
                int(total_len),
                int(self.quantizer.mse_quantizer.bits),
                int(num_heads),
                int(kv_heads),
            )
        return scores.view(batch, num_heads, q_len, total_len)

    def av(
        self,
        attn_weights: torch.Tensor,
        total_len: int,
        num_heads: int,
        kv_heads: int,
        profiler: TurboQuantProfiler,
    ) -> torch.Tensor:
        batch, _, q_len, _ = attn_weights.shape
        assert q_len == 1
        attn_f32 = attn_weights if attn_weights.dtype == torch.float32 else attn_weights.to(torch.float32)
        with profiler.scope("quant_av_prod", attn_weights.device):
            mse_rotated = _ext.av_mse_rotated(
                attn_f32.reshape(batch * num_heads, q_len, total_len).contiguous(),
                self._packed_mse,
                self._norms_f32[:, :, 0].contiguous(),
                self.quantizer.mse_quantizer.centroids.contiguous(),
                int(total_len),
                int(self.quantizer.mse_quantizer.bits),
                int(num_heads),
                int(kv_heads),
            )
            qjl_rotated = _ext.av_mse_rotated(
                attn_f32.reshape(batch * num_heads, q_len, total_len).contiguous(),
                self._qjl,
                self._residual_norms_f32[:, :, 0].contiguous(),
                self._qjl_centroids,
                int(total_len),
                1,
                int(num_heads),
                int(kv_heads),
            )
            output = torch.matmul(mse_rotated[..., : self.quantizer.dim], self.quantizer.mse_quantizer.rotation)
            output.add_(self.quantizer.coeff * torch.matmul(qjl_rotated[..., : self.quantizer.dim], self.quantizer.S))
        return output.view(batch, num_heads, q_len, self.quantizer.dim).to(attn_weights.dtype)

    def write_range_to(self, file: CacheFile, offset: int, start: int, end: int) -> None:
        _write_token_range(file, self._packed_mse, offset, start, end)
        qjl_offset = offset + self._packed_mse.nbytes
        _write_token_range(file, self._qjl, qjl_offset, start, end)
        residual_norm_offset = qjl_offset + self._qjl.nbytes
        _write_token_range(file, self._residual_norms, residual_norm_offset, start, end)
        norms_offset = residual_norm_offset + self._residual_norms.nbytes
        _write_token_range(file, self._norms, norms_offset, start, end)

    def read_from(self, file: CacheFile, offset: int) -> None:
        _read_token_range(file, self._packed_mse, offset)
        qjl_offset = offset + self._packed_mse.nbytes
        _read_token_range(file, self._qjl, qjl_offset)
        residual_norm_offset = qjl_offset + self._qjl.nbytes
        _read_token_range(file, self._residual_norms, residual_norm_offset)
        norms_offset = residual_norm_offset + self._residual_norms.nbytes
        _read_token_range(file, self._norms, norms_offset)
        self._residual_norms_f32.copy_(self._residual_norms.to(torch.float32))
        self._norms_f32.copy_(self._norms.to(torch.float32))

    def append_token_buffers(self, buffers: list[bytes], token: int) -> None:
        _append_tensor_token_bytes(buffers, self._packed_mse, token)
        _append_tensor_token_bytes(buffers, self._qjl, token)
        _append_tensor_token_bytes(buffers, self._residual_norms, token)
        _append_tensor_token_bytes(buffers, self._norms, token)

    def pack_token_record(self, record: torch.Tensor, token: int, offset: int) -> int:
        _ext.pack_token_slice(self._packed_mse, record, int(token), int(offset))
        offset += _token_slice_nbytes(self._packed_mse)
        _ext.pack_token_slice(self._qjl, record, int(token), int(offset))
        offset += _token_slice_nbytes(self._qjl)
        _ext.pack_token_slice(self._residual_norms, record, int(token), int(offset))
        offset += _token_slice_nbytes(self._residual_norms)
        _ext.pack_token_slice(self._norms, record, int(token), int(offset))
        offset += _token_slice_nbytes(self._norms)
        return offset

    def load_tokens(self, raw: bytearray, start_token: int, num_tokens: int, record_nbytes: int, offset: int) -> int:
        end_token = start_token + num_tokens
        offset = _load_tensor_tokens(raw, self._packed_mse, start_token, num_tokens, record_nbytes, offset)
        offset = _load_tensor_tokens(raw, self._qjl, start_token, num_tokens, record_nbytes, offset)
        offset = _load_tensor_tokens(raw, self._residual_norms, start_token, num_tokens, record_nbytes, offset)
        offset = _load_tensor_tokens(raw, self._norms, start_token, num_tokens, record_nbytes, offset)
        self._residual_norms_f32[:, start_token:end_token, :].copy_(
            self._residual_norms[:, start_token:end_token, :].to(torch.float32)
        )
        self._norms_f32[:, start_token:end_token, :].copy_(self._norms[:, start_token:end_token, :].to(torch.float32))
        return offset


@dataclass
class TurboQuantConfig:
    regular_bits: int
    outlier_bits: int
    outlier_channels: int
    seed: int
    mode: str
    device: torch.device
    kv_heads: int
    head_dim: int
    max_len: int
    profile: bool
    sync_timers: bool


class TurboQuantProfiler:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def add(self, name: str, duration_ms: float) -> None:
        if not self.enabled:
            return
        self._totals[name] = self._totals.get(name, 0.0) + duration_ms
        self._counts[name] = self._counts.get(name, 0) + 1

    def stats(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "total_ms": total_ms,
                "count": self._counts[name],
                "avg_ms": total_ms / self._counts[name],
            }
            for name, total_ms in self._totals.items()
        }

    def scope(self, name: str, device: torch.device):
        return _TurboQuantProfileScope(self, name, device)


class _TurboQuantProfileScope:
    def __init__(self, profiler: TurboQuantProfiler, name: str, device: torch.device):
        self.profiler = profiler
        self.name = name
        self.device = device
        self.start = 0.0

    def __enter__(self):
        if not self.profiler.enabled:
            return self
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.profiler.enabled:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.profiler.add(self.name, (perf_counter() - self.start) * 1000.0)


def _build_key_store(config: TurboQuantConfig, dim: int, bits: int, seed: int):
    if config.mode == "mse":
        return MSETokenStore(
            quantizer=TurboQuantMSEQuantizer(dim, bits, seed, config.device),
            kv_heads=config.kv_heads,
            max_len=config.max_len,
            device=config.device,
        )
    if config.mode == "prod":
        assert bits >= 2
        return ProdTokenStore(
            quantizer=TurboQuantProdQuantizer(dim, bits, seed, config.device),
            kv_heads=config.kv_heads,
            max_len=config.max_len,
            device=config.device,
        )
    raise ValueError(f"unsupported quantization mode: {config.mode}")


def _build_value_store(config: TurboQuantConfig, dim: int, bits: int, seed: int):
    if config.mode in {"mse", "prod"}:
        return MSETokenStore(
            quantizer=TurboQuantMSEQuantizer(dim, bits, seed, config.device),
            kv_heads=config.kv_heads,
            max_len=config.max_len,
            device=config.device,
        )
    raise ValueError(f"unsupported quantization mode: {config.mode}")


def _token_major_offset(tensor: torch.Tensor, base_offset: int, start: int) -> int:
    """Compute byte offset for token `start` in a token-major layout of [heads, max_len, dim]."""
    _, _, inner_dim = tensor.shape
    num_heads = tensor.shape[0]
    token_bytes = num_heads * inner_dim * tensor.element_size()
    return base_offset + start * token_bytes


def _write_token_range(file: CacheFile, tensor: torch.Tensor, offset: int, start: int, end: int) -> None:
    """Write tokens [start, end) in token-major layout: [max_len, heads, dim] on disk."""
    assert end > start
    # tensor is [heads, max_len, dim] -> permute to [max_len, heads, dim], slice, then write contiguously
    token_offset = _token_major_offset(tensor, offset, start)
    file.write_tensor(tensor[:, start:end, :].permute(1, 0, 2), token_offset)


def _read_token_range(file: CacheFile, tensor: torch.Tensor, offset: int) -> None:
    """Read the full tensor from token-major layout on disk back into [heads, max_len, dim]."""
    token_major = tensor.permute(1, 0, 2).contiguous()
    file.read_tensor(token_major, token_major.nbytes, offset)
    tensor.copy_(token_major.permute(1, 0, 2))


def _token_slice_nbytes(tensor: torch.Tensor) -> int:
    return tensor.shape[0] * tensor.shape[2] * tensor.element_size()


def _append_tensor_token_bytes(buffers: list[bytes], tensor: torch.Tensor, token: int) -> None:
    buffers.append(tensor[:, token : token + 1, :].contiguous().cpu().view(torch.uint8).numpy().tobytes())


def _load_tensor_tokens(
    raw: bytearray,
    tensor: torch.Tensor,
    start_token: int,
    num_tokens: int,
    record_nbytes: int,
    offset: int,
) -> int:
    token_nbytes = _token_slice_nbytes(tensor)
    element_size = tensor.element_size()
    record_stride = record_nbytes // element_size
    storage_offset = offset // element_size
    raw_tensor = torch.frombuffer(raw, dtype=tensor.dtype)
    token_major = torch.as_strided(
        raw_tensor,
        (num_tokens, tensor.shape[0], tensor.shape[2]),
        (record_stride, tensor.shape[2], 1),
        storage_offset=storage_offset,
    )
    tensor[:, start_token : start_token + num_tokens, :].copy_(token_major.permute(1, 0, 2).to(tensor.device))
    return offset + token_nbytes


class LayerTurboQuantCache:
    def __init__(self, config: TurboQuantConfig, layer_idx: int, profiler: TurboQuantProfiler):
        self.config = config
        self.profiler = profiler
        self.total_len = 0
        self.saved_len = 0
        self.outlier_idx = torch.empty(0, dtype=torch.long, device=config.device)
        self.regular_idx = torch.empty(0, dtype=torch.long, device=config.device)
        base_seed = config.seed + layer_idx * 1000
        regular_dim, outlier_dim = _split_dims(config.head_dim, config.outlier_channels)
        self.key_regular = _build_key_store(config, regular_dim, config.regular_bits, base_seed + 11)
        self.key_outlier = _build_key_store(config, outlier_dim, config.outlier_bits, base_seed + 17)
        self.value_regular = _build_value_store(config, regular_dim, config.regular_bits, base_seed + 23)
        self.value_outlier = _build_value_store(config, outlier_dim, config.outlier_bits, base_seed + 29)

    def _assign_split_indices(self, outlier_idx: torch.Tensor) -> None:
        """Build regular/outlier channel index tensors once the outlier channels are known."""
        self.outlier_idx = outlier_idx.to(device=self.config.device, dtype=torch.long)
        regular_mask = torch.ones(self.config.head_dim, dtype=torch.bool, device=self.config.device)
        regular_mask[self.outlier_idx] = False
        self.regular_idx = regular_mask.nonzero(as_tuple=False).flatten()
        # int32 versions for CUDA kernels (append_kv_mse_split expects int32)
        self.regular_idx_i32 = self.regular_idx.to(torch.int32)
        self.outlier_idx_i32 = self.outlier_idx.to(torch.int32)

    def _init_split(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        score = key_states[0].pow(2).mean(dim=(0, 1)) + value_states[0].pow(2).mean(dim=(0, 1))
        self._assign_split_indices(torch.topk(score, k=self.config.outlier_channels).indices.sort().values)

    def update_range(self, start: int, end: int, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        assert start == self.total_len, f"only append supported: start={start} != total_len={self.total_len}"
        if self.total_len == 0:
            self._init_split(key_states, value_states)
        if self.config.mode == "prod":
            key = key_states[0].to(torch.bfloat16).contiguous()
            value = value_states[0].to(torch.bfloat16).contiguous()
            self.key_regular.write(start, key.index_select(-1, self.regular_idx))
            self.key_outlier.write(start, key.index_select(-1, self.outlier_idx))
            self.value_regular.write(start, value.index_select(-1, self.regular_idx))
            self.value_outlier.write(start, value.index_select(-1, self.outlier_idx))
            self.total_len = end
            return

        # key_states/value_states: (1, kv_heads, token_len, head_dim) -> (kv_heads, token_len, head_dim)
        key = key_states[0].to(torch.bfloat16).contiguous()
        value = value_states[0].to(torch.bfloat16).contiguous()
        kr, ko = self.key_regular, self.key_outlier
        vr, vo = self.value_regular, self.value_outlier
        _ext.append_kv_mse_split(
            key,
            value,
            self.regular_idx_i32,
            self.outlier_idx_i32,
            kr.quantizer.rotation_t,
            ko.quantizer.rotation_t,
            kr.quantizer.boundaries.contiguous(),
            ko.quantizer.boundaries.contiguous(),
            vr.quantizer.rotation_t,
            vo.quantizer.rotation_t,
            vr.quantizer.boundaries.contiguous(),
            vo.quantizer.boundaries.contiguous(),
            kr._packed,
            kr._norms[:, :, 0],
            kr._norms_f32[:, :, 0],
            ko._packed,
            ko._norms[:, :, 0],
            ko._norms_f32[:, :, 0],
            vr._packed,
            vr._norms[:, :, 0],
            vr._norms_f32[:, :, 0],
            vo._packed,
            vo._norms[:, :, 0],
            vo._norms_f32[:, :, 0],
            int(start),
            int(kr.quantizer.bits),
            int(ko.quantizer.bits),
        )
        self.total_len = end

    def restore(self, metadata: dict) -> None:
        total_len = metadata["total_len"]
        outlier_idx = metadata["outlier_idx"]
        self.total_len = total_len
        self.saved_len = total_len
        self._assign_split_indices(torch.tensor(outlier_idx, dtype=torch.long))

    def get_metadata(self) -> dict:
        return {
            "total_len": self.total_len,
            "outlier_idx": self.outlier_idx.tolist(),
        }

    def _dequantize(self, regular_store, outlier_store, length: int, include_residual: bool = True) -> torch.Tensor:
        full = torch.empty(
            (self.config.kv_heads, length, self.config.head_dim), dtype=TURBOQUANT_DTYPE, device=self.config.device
        )
        full.index_copy_(-1, self.regular_idx, regular_store.dequantize(length, include_residual=include_residual))
        full.index_copy_(-1, self.outlier_idx, outlier_store.dequantize(length, include_residual=include_residual))
        return full

    def dequantize_keys(self) -> torch.Tensor:
        return self._dequantize(self.key_regular, self.key_outlier, self.total_len)

    def dequantize_values(self) -> torch.Tensor:
        return self._dequantize(self.value_regular, self.value_outlier, self.total_len)

    def key_nbytes(self) -> int:
        return self.key_regular.nbytes + self.key_outlier.nbytes

    def value_nbytes(self) -> int:
        return self.value_regular.nbytes + self.value_outlier.nbytes

    def decode_record_nbytes(self) -> int:
        return (
            self.key_regular.token_nbytes
            + self.key_outlier.token_nbytes
            + self.value_regular.token_nbytes
            + self.value_outlier.token_nbytes
        )

    def write_key_range_to(self, file: CacheFile, offset: int, start: int, end: int) -> None:
        self.key_regular.write_range_to(file, offset, start, end)
        self.key_outlier.write_range_to(file, offset + self.key_regular.nbytes, start, end)

    def write_value_range_to(self, file: CacheFile, offset: int, start: int, end: int) -> None:
        self.value_regular.write_range_to(file, offset, start, end)
        self.value_outlier.write_range_to(file, offset + self.value_regular.nbytes, start, end)

    def read_key_from(self, file: CacheFile, offset: int) -> None:
        self.key_regular.read_from(file, offset)
        self.key_outlier.read_from(file, offset + self.key_regular.nbytes)

    def read_value_from(self, file: CacheFile, offset: int) -> None:
        self.value_regular.read_from(file, offset)
        self.value_outlier.read_from(file, offset + self.value_regular.nbytes)

    def append_decode_token_buffers(self, buffers: list[bytes], token: int) -> None:
        self.key_regular.append_token_buffers(buffers, token)
        self.value_regular.append_token_buffers(buffers, token)
        self.key_outlier.append_token_buffers(buffers, token)
        self.value_outlier.append_token_buffers(buffers, token)

    def pack_decode_token_record(self, record: torch.Tensor, token: int, offset: int) -> int:
        offset = self.key_regular.pack_token_record(record, token, offset)
        offset = self.value_regular.pack_token_record(record, token, offset)
        offset = self.key_outlier.pack_token_record(record, token, offset)
        offset = self.value_outlier.pack_token_record(record, token, offset)
        return offset

    def load_decode_tokens(
        self,
        raw: bytearray,
        start_token: int,
        num_tokens: int,
        record_nbytes: int,
        offset: int,
    ) -> int:
        if self.total_len == 0:
            raise AssertionError("decode token load requires prefix to initialize outlier split")
        offset = self.key_regular.load_tokens(raw, start_token, num_tokens, record_nbytes, offset)
        offset = self.value_regular.load_tokens(raw, start_token, num_tokens, record_nbytes, offset)
        offset = self.key_outlier.load_tokens(raw, start_token, num_tokens, record_nbytes, offset)
        offset = self.value_outlier.load_tokens(raw, start_token, num_tokens, record_nbytes, offset)
        self.total_len = max(self.total_len, start_token + num_tokens)
        self.saved_len = max(self.saved_len, start_token + num_tokens)
        return offset


class TurboQuantKVCache(BaseKVCache):
    METADATA_FILE = "turboquant_layers.json"

    @staticmethod
    def compute_file_sizes(*, config: ModelShape, runtime_config: RuntimeConfig, max_len: int) -> tuple[int, int]:
        tc = config.text_config
        bits = runtime_config.turboquant_bits
        outlier_bits = runtime_config.turboquant_outlier_bits
        regular_dim, outlier_dim = _split_dims(tc.head_dim, runtime_config.turboquant_outlier_channels)
        key_per_layer = _store_nbytes(runtime_config, tc, max_len, regular_dim, bits, kind="key")
        key_per_layer += _store_nbytes(runtime_config, tc, max_len, outlier_dim, outlier_bits, kind="key")
        value_per_layer = _store_nbytes(runtime_config, tc, max_len, regular_dim, bits, kind="value")
        value_per_layer += _store_nbytes(runtime_config, tc, max_len, outlier_dim, outlier_bits, kind="value")
        return tc.num_hidden_layers * key_per_layer, tc.num_hidden_layers * value_per_layer

    @staticmethod
    def build(
        *,
        config: ModelShape,
        runtime_config: RuntimeConfig,
        max_len: int,
        cache_metadata: dict | None = None,
    ) -> TurboQuantKVCache:
        tc = config.text_config
        device = torch.device(runtime_config.device)
        return TurboQuantKVCache(
            num_layers=tc.num_hidden_layers,
            cache_metadata=cache_metadata,
            config=TurboQuantConfig(
                regular_bits=runtime_config.turboquant_bits,
                outlier_bits=runtime_config.turboquant_outlier_bits,
                outlier_channels=runtime_config.turboquant_outlier_channels,
                seed=runtime_config.turboquant_seed,
                mode=runtime_config.turboquant_mode,
                device=device,
                kv_heads=tc.num_key_value_heads,
                head_dim=tc.head_dim,
                max_len=max_len,
                profile=runtime_config.turboquant_profile,
                sync_timers=runtime_config.sync_cuda_timers,
            ),
        )

    def __init__(self, *, num_layers: int, config: TurboQuantConfig, cache_metadata: dict | None = None):
        self.profiler = TurboQuantProfiler(config.profile)
        self.layers = [LayerTurboQuantCache(config, layer_idx, self.profiler) for layer_idx in range(num_layers)]
        self._decode_record_buffer = torch.empty(self.decode_record_nbytes(), dtype=torch.uint8, device=config.device)
        self._decode_base_len = cache_metadata["decode_base_len"] if cache_metadata else None
        self._decode_prefill_layers = cache_metadata.get("layers") if cache_metadata else None
        self._key_layer_offsets = []
        self._value_layer_offsets = []
        key_offset = 0
        value_offset = 0
        for layer in self.layers:
            self._key_layer_offsets.append(key_offset)
            self._value_layer_offsets.append(value_offset)
            key_offset += layer.key_nbytes()
            value_offset += layer.value_nbytes()
        self._key_file_size = key_offset
        self._value_file_size = value_offset
        if cache_metadata and "layers" in cache_metadata:
            for layer, meta in enumerate(cache_metadata["layers"]):
                self.layers[layer].restore(meta)

    @classmethod
    def from_dense_cache(cls, state: GenerationState, dense_cache: DenseKVCache) -> TurboQuantKVCache:
        from rover.runtime.types import Range

        quantized_cache = cls.build(
            config=state.config,
            runtime_config=state.runtime_config,
            max_len=state.max_len,
        )
        prompt_len = state.prompt_len
        for layer in range(state.num_layers):
            key_states, value_states = dense_cache.get_kv(Range(layer, 0, prompt_len))
            quantized_cache.update_kv(Range(layer, 0, prompt_len), key_states, value_states)
        return quantized_cache

    def cache_format(self) -> str:
        return "turboquant"

    def update_kv(self, r: Range, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        layer_cache = self.layers[r.layer]
        with self.profiler.scope("decode_quantize_write", key_states.device):
            layer_cache.update_range(r.start, r.end, key_states, value_states)

    def get_kv_for_attention(self, layer: int) -> LayerTurboQuantCache:
        return self.layers[layer]

    def get_metadata(self) -> dict:
        return {
            "decode_base_len": self._decode_base_len,
        }

    def metadata_payload(self, base_dir: Path) -> dict:
        layers = self._decode_prefill_layers
        if layers is None:
            layers = [layer.get_metadata() for layer in self.layers]
        sidecar = base_dir / self.METADATA_FILE
        if self._decode_prefill_layers is None or not sidecar.exists():
            with open(sidecar, "w") as f:
                json.dump({"layers": layers}, f)
        return {
            "decode_base_len": self._decode_base_len,
            "layers_file": self.METADATA_FILE,
        }

    @classmethod
    def resolve_metadata_payload(cls, base_dir: Path, payload: dict) -> dict:
        if "layers_file" not in payload:
            return payload
        with open(base_dir / payload["layers_file"]) as f:
            return {**payload, **json.load(f)}

    def prefill_key_file_size(self) -> int:
        return self._key_file_size

    def prefill_value_file_size(self) -> int:
        return self._value_file_size

    def _key_layer_offset(self, layer: int) -> int:
        return self._key_layer_offsets[layer]

    def _value_layer_offset(self, layer: int) -> int:
        return self._value_layer_offsets[layer]

    def profile_stats(self) -> dict[str, dict[str, float | int]]:
        return self.profiler.stats()

    def save_prefill_layer(self, layer: int, start: int, end: int, k_file: CacheFile, v_file: CacheFile) -> None:
        lc = self.layers[layer]
        if end <= start:
            return
        lc.write_key_range_to(k_file, self._key_layer_offset(layer), start, end)
        lc.write_value_range_to(v_file, self._value_layer_offset(layer), start, end)
        lc.saved_len = max(lc.saved_len, end)

    def load_prefill_layer(self, layer: int, start: int, end: int, k_file: CacheFile, v_file: CacheFile) -> None:
        lc = self.layers[layer]
        assert lc.outlier_idx.numel() == lc.config.outlier_channels
        lc.read_key_from(k_file, self._key_layer_offset(layer))
        lc.read_value_from(v_file, self._value_layer_offset(layer))
        lc.saved_len = max(lc.saved_len, end)

    def begin_decode(self, prefix_len: int) -> None:
        self._decode_base_len = prefix_len
        self._decode_prefill_layers = [layer.get_metadata() for layer in self.layers]

    def decode_record_nbytes(self) -> int:
        return sum(layer.decode_record_nbytes() for layer in self.layers)

    def save_decode_tokens(self, start_token: int, end_token: int, decode_file: CacheFile) -> None:
        assert self._decode_base_len is not None
        assert end_token >= start_token
        record_nbytes = self.decode_record_nbytes()
        sync_cuda = self.layers[0].config.sync_timers
        for token in range(start_token, end_token):
            absolute_token = self._decode_base_len + token
            with tracer.timer("commit_save_kv_pack", cache_format="turboquant", sync_cuda=sync_cuda):
                offset = 0
                for layer in self.layers:
                    offset = layer.pack_decode_token_record(self._decode_record_buffer, absolute_token, offset)
            with tracer.timer("commit_save_kv_d2h", cache_format="turboquant", sync_cuda=sync_cuda):
                record = self._decode_record_buffer.cpu()
            with tracer.timer("commit_save_kv_write", cache_format="turboquant", sync_cuda=sync_cuda):
                decode_file.write(memoryview(record.numpy()), token * record_nbytes)

    def load_decode_tokens(self, decode_file: CacheFile, num_tokens: int) -> None:
        assert self._decode_base_len is not None
        record_nbytes = self.decode_record_nbytes()
        if num_tokens <= 0:
            return
        raw = bytearray(decode_file.read(record_nbytes * num_tokens, 0))
        start_token = self._decode_base_len
        offset = 0
        for layer in self.layers:
            offset = layer.load_decode_tokens(raw, start_token, num_tokens, record_nbytes, offset)


def turboquant_attention_forward(
    query: torch.Tensor,
    layer_cache: LayerTurboQuantCache,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch, num_heads, q_len, head_dim = query.shape
    assert q_len == 1
    kv_heads = layer_cache.config.kv_heads
    profiler = layer_cache.profiler
    query_f32 = query.to(torch.float32)
    sync_cuda = layer_cache.config.sync_timers and query.device.type == "cuda"
    with tracer.timer("decode_attn_total", cache_format="turboquant", sync_cuda=sync_cuda):
        with tracer.timer("decode_attn_qk", cache_format="turboquant", sync_cuda=sync_cuda):
            if layer_cache.config.mode == "mse":
                kr, ko = layer_cache.key_regular, layer_cache.key_outlier
                attn_scores = _ext.qk_mse_split(
                    query_f32.reshape(batch * num_heads, q_len, head_dim),
                    layer_cache.regular_idx_i32,
                    layer_cache.outlier_idx_i32,
                    kr.quantizer.rotation_t,
                    ko.quantizer.rotation_t,
                    kr._packed,
                    kr._norms_f32[:, :, 0].contiguous(),
                    kr.quantizer.centroids.contiguous(),
                    ko._packed,
                    ko._norms_f32[:, :, 0].contiguous(),
                    ko.quantizer.centroids.contiguous(),
                    int(layer_cache.total_len),
                    int(kr.quantizer.bits),
                    int(ko.quantizer.bits),
                    int(num_heads),
                    int(kv_heads),
                ).view(batch, num_heads, q_len, layer_cache.total_len)
            else:
                attn_scores = layer_cache.key_regular.qk(
                    query_f32.index_select(-1, layer_cache.regular_idx),
                    layer_cache.total_len,
                    num_heads,
                    kv_heads,
                    profiler,
                )
                attn_scores.add_(
                    layer_cache.key_outlier.qk(
                        query_f32.index_select(-1, layer_cache.outlier_idx),
                        layer_cache.total_len,
                        num_heads,
                        kv_heads,
                        profiler,
                    )
                )

        with tracer.timer("decode_attn_softmax", cache_format="turboquant", sync_cuda=sync_cuda):
            with profiler.scope("quant_softmax", query.device):
                attn_scores.mul_(scaling)
                if attention_mask is not None:
                    attn_scores.add_(attention_mask)
                attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32)

        with tracer.timer("decode_attn_av", cache_format="turboquant", sync_cuda=sync_cuda):
            if layer_cache.config.mode == "mse":
                vr, vo = layer_cache.value_regular, layer_cache.value_outlier
                attn_output = _ext.av_mse_split(
                    attn_weights.reshape(batch * num_heads, q_len, layer_cache.total_len).contiguous(),
                    layer_cache.regular_idx,
                    layer_cache.outlier_idx,
                    vr._packed,
                    vr._norms_f32[:, :, 0].contiguous(),
                    vr.quantizer.centroids.contiguous(),
                    vr.quantizer.rotation,
                    vo._packed,
                    vo._norms_f32[:, :, 0].contiguous(),
                    vo.quantizer.centroids.contiguous(),
                    vo.quantizer.rotation,
                    int(layer_cache.total_len),
                    int(vr.quantizer.bits),
                    int(vo.quantizer.bits),
                    int(num_heads),
                    int(kv_heads),
                    int(head_dim),
                ).view(batch, num_heads, q_len, head_dim)
            else:
                attn_output = torch.empty((batch, num_heads, q_len, head_dim), dtype=torch.float32, device=query.device)
                attn_output.zero_()
                regular_out = layer_cache.value_regular.av(
                    attn_weights,
                    layer_cache.total_len,
                    num_heads,
                    kv_heads,
                    profiler,
                )
                attn_output.index_copy_(-1, layer_cache.regular_idx, regular_out)
                outlier_out = layer_cache.value_outlier.av(
                    attn_weights,
                    layer_cache.total_len,
                    num_heads,
                    kv_heads,
                    profiler,
                )
                attn_output.index_copy_(-1, layer_cache.outlier_idx, outlier_out)

        with tracer.timer("decode_attn_output", cache_format="turboquant", sync_cuda=sync_cuda):
            attn_output = (
                attn_output.to(query.dtype).transpose(1, 2).contiguous().reshape(batch, q_len, num_heads * head_dim)
            )
    return attn_output, attn_weights


@dataclass
class TurboQuantGraphBuffers:
    """Pre-allocated buffers for graph-safe turboquant attention."""

    q_reg: torch.Tensor
    q_out: torch.Tensor
    q_reg_rot: torch.Tensor
    q_out_rot: torch.Tensor
    scores: torch.Tensor
    reg_rotated: torch.Tensor
    out_rotated: torch.Tensor
    reg_out: torch.Tensor
    out_out: torch.Tensor
    av_output: torch.Tensor
    prod_key_regular: ProdTokenStore | None = None
    prod_key_outlier: ProdTokenStore | None = None
    prod_value_regular: MSETokenStore | None = None
    prod_value_outlier: MSETokenStore | None = None

    @staticmethod
    def create(layer_cache: LayerTurboQuantCache, num_heads: int, max_len: int) -> TurboQuantGraphBuffers:
        device = layer_cache.config.device
        reg_dim = int(layer_cache.regular_idx.numel())
        out_dim = int(layer_cache.outlier_idx.numel())
        if layer_cache.config.mode == "mse":
            vr, vo = layer_cache.value_regular, layer_cache.value_outlier
            reg_packed_D = vr._packed.size(2)
            out_packed_D = vo._packed.size(2)
            reg_D = (32 // vr.quantizer.bits) * reg_packed_D
            out_D = (32 // vo.quantizer.bits) * out_packed_D
            prod_key_regular = None
            prod_key_outlier = None
            prod_value_regular = None
            prod_value_outlier = None
            q_reg_dim = reg_dim
            q_out_dim = out_dim
        else:
            kr, ko = layer_cache.key_regular, layer_cache.key_outlier
            vr, vo = layer_cache.value_regular, layer_cache.value_outlier
            reg_D = reg_dim
            out_D = out_dim
            prod_key_regular = ProdTokenStore(
                quantizer=kr.quantizer, kv_heads=layer_cache.config.kv_heads, max_len=1, device=device
            )
            prod_key_outlier = ProdTokenStore(
                quantizer=ko.quantizer, kv_heads=layer_cache.config.kv_heads, max_len=1, device=device
            )
            prod_value_regular = MSETokenStore(
                quantizer=vr.quantizer, kv_heads=layer_cache.config.kv_heads, max_len=1, device=device
            )
            prod_value_outlier = MSETokenStore(
                quantizer=vo.quantizer, kv_heads=layer_cache.config.kv_heads, max_len=1, device=device
            )
            q_reg_dim = reg_dim
            q_out_dim = out_dim
        return TurboQuantGraphBuffers(
            q_reg=torch.empty(num_heads, 1, q_reg_dim, dtype=torch.float32, device=device),
            q_out=torch.empty(num_heads, 1, q_out_dim, dtype=torch.float32, device=device),
            q_reg_rot=torch.empty(num_heads, 1, q_reg_dim, dtype=torch.float32, device=device),
            q_out_rot=torch.empty(num_heads, 1, q_out_dim, dtype=torch.float32, device=device),
            scores=torch.zeros(num_heads, 1, max_len, dtype=torch.float32, device=device),
            reg_rotated=torch.empty(num_heads, 1, reg_D, dtype=torch.float32, device=device),
            out_rotated=torch.empty(num_heads, 1, out_D, dtype=torch.float32, device=device),
            reg_out=torch.empty(num_heads, 1, reg_dim, dtype=torch.float32, device=device),
            out_out=torch.empty(num_heads, 1, out_dim, dtype=torch.float32, device=device),
            av_output=torch.zeros(num_heads, 1, layer_cache.config.head_dim, dtype=torch.float32, device=device),
            prod_key_regular=prod_key_regular,
            prod_key_outlier=prod_key_outlier,
            prod_value_regular=prod_value_regular,
            prod_value_outlier=prod_value_outlier,
        )


def _graph_write_store(
    store: ProdTokenStore | MSETokenStore,
    temp_store: ProdTokenStore | MSETokenStore,
    x: torch.Tensor,
    start_index_tensor: torch.Tensor,
) -> None:
    temp_store.write(0, x)
    if isinstance(store, ProdTokenStore):
        assert isinstance(temp_store, ProdTokenStore)
        store._packed_mse.index_copy_(1, start_index_tensor, temp_store._packed_mse)
        store._qjl.index_copy_(1, start_index_tensor, temp_store._qjl)
        store._residual_norms.index_copy_(1, start_index_tensor, temp_store._residual_norms)
        store._residual_norms_f32.index_copy_(1, start_index_tensor, temp_store._residual_norms_f32)
        store._norms.index_copy_(1, start_index_tensor, temp_store._norms)
        store._norms_f32.index_copy_(1, start_index_tensor, temp_store._norms_f32)
        return
    assert isinstance(temp_store, MSETokenStore)
    store._packed.index_copy_(1, start_index_tensor, temp_store._packed)
    store._norms.index_copy_(1, start_index_tensor, temp_store._norms)
    store._norms_f32.index_copy_(1, start_index_tensor, temp_store._norms_f32)


def _turboquant_attention_graph_forward_prod(
    query: torch.Tensor,
    layer_cache: LayerTurboQuantCache,
    attention_mask: torch.Tensor,
    scaling: float,
    bufs: TurboQuantGraphBuffers,
    max_len: int,
) -> torch.Tensor:
    batch, num_heads, q_len, head_dim = query.shape
    kv_heads = layer_cache.config.kv_heads
    query_f32 = query.to(torch.float32)

    torch.index_select(
        query_f32.reshape(batch * num_heads, q_len, head_dim), -1, layer_cache.regular_idx, out=bufs.q_reg
    )
    torch.index_select(
        query_f32.reshape(batch * num_heads, q_len, head_dim), -1, layer_cache.outlier_idx, out=bufs.q_out
    )

    bufs.scores.copy_(
        layer_cache.key_regular.qk(
            bufs.q_reg.view(batch, num_heads, q_len, -1),
            max_len,
            num_heads,
            kv_heads,
            layer_cache.profiler,
        ).view(batch * num_heads, q_len, max_len)
    )
    bufs.scores.add_(
        layer_cache.key_outlier.qk(
            bufs.q_out.view(batch, num_heads, q_len, -1),
            max_len,
            num_heads,
            kv_heads,
            layer_cache.profiler,
        ).view(batch * num_heads, q_len, max_len)
    )

    attn_scores = bufs.scores.view(batch, num_heads, q_len, max_len)
    attn_scores.mul_(scaling)
    attn_scores.add_(attention_mask)
    attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32)
    bufs.av_output.zero_()
    bufs.reg_out.copy_(
        layer_cache.value_regular.av(attn_weights, max_len, num_heads, kv_heads, layer_cache.profiler).view(
            batch * num_heads, q_len, -1
        )
    )
    bufs.av_output.index_copy_(-1, layer_cache.regular_idx, bufs.reg_out)
    bufs.out_out.copy_(
        layer_cache.value_outlier.av(attn_weights, max_len, num_heads, kv_heads, layer_cache.profiler).view(
            batch * num_heads, q_len, -1
        )
    )
    bufs.av_output.index_copy_(-1, layer_cache.outlier_idx, bufs.out_out)
    return (
        bufs.av_output.view(batch, num_heads, q_len, head_dim)
        .to(query.dtype)
        .transpose(1, 2)
        .contiguous()
        .reshape(batch, q_len, num_heads * head_dim)
    )


def turboquant_attention_graph_forward(
    query: torch.Tensor,
    layer_cache: LayerTurboQuantCache,
    attention_mask: torch.Tensor,
    scaling: float,
    bufs: TurboQuantGraphBuffers,
    total_len_tensor: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    """Graph-safe turboquant attention using pre-allocated buffers."""
    if layer_cache.config.mode == "prod":
        return _turboquant_attention_graph_forward_prod(query, layer_cache, attention_mask, scaling, bufs, max_len)

    batch, num_heads, q_len, head_dim = query.shape
    kv_heads = layer_cache.config.kv_heads
    query_f32 = query.to(torch.float32).reshape(batch * num_heads, q_len, head_dim)

    kr, ko = layer_cache.key_regular, layer_cache.key_outlier
    torch.index_select(query_f32, -1, layer_cache.regular_idx, out=bufs.q_reg)
    torch.index_select(query_f32, -1, layer_cache.outlier_idx, out=bufs.q_out)
    torch.matmul(bufs.q_reg, kr.quantizer.rotation_t, out=bufs.q_reg_rot)
    torch.matmul(bufs.q_out, ko.quantizer.rotation_t, out=bufs.q_out_rot)
    _ext.qk_mse_split_graph_rotated(
        bufs.q_reg_rot,
        bufs.q_out_rot,
        kr._packed,
        kr._norms_f32[:, :, 0].contiguous(),
        kr.quantizer.centroids.contiguous(),
        ko._packed,
        ko._norms_f32[:, :, 0].contiguous(),
        ko.quantizer.centroids.contiguous(),
        bufs.scores,
        total_len_tensor,
        max_len,
        int(kr.quantizer.bits),
        int(ko.quantizer.bits),
        int(num_heads),
        int(kv_heads),
    )

    attn_scores = bufs.scores.view(batch, num_heads, q_len, max_len)
    attn_scores.mul_(scaling)
    attn_scores.add_(attention_mask)
    attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32)

    vr, vo = layer_cache.value_regular, layer_cache.value_outlier
    _ext.av_mse_split_graph_rotated(
        attn_weights.reshape(batch * num_heads, q_len, max_len).contiguous(),
        vr._packed,
        vr._norms_f32[:, :, 0].contiguous(),
        vr.quantizer.centroids.contiguous(),
        vo._packed,
        vo._norms_f32[:, :, 0].contiguous(),
        vo.quantizer.centroids.contiguous(),
        bufs.reg_rotated,
        bufs.out_rotated,
        total_len_tensor,
        max_len,
        int(vr.quantizer.bits),
        int(vo.quantizer.bits),
        int(num_heads),
        int(kv_heads),
    )
    reg_dim = int(layer_cache.regular_idx.numel())
    out_dim = int(layer_cache.outlier_idx.numel())
    torch.matmul(bufs.reg_rotated[..., :reg_dim], vr.quantizer.rotation, out=bufs.reg_out)
    torch.matmul(bufs.out_rotated[..., :out_dim], vo.quantizer.rotation, out=bufs.out_out)
    bufs.av_output.zero_()
    bufs.av_output.index_copy_(-1, layer_cache.regular_idx, bufs.reg_out)
    bufs.av_output.index_copy_(-1, layer_cache.outlier_idx, bufs.out_out)

    attn_output = bufs.av_output.view(batch, num_heads, q_len, head_dim)
    return attn_output.to(query.dtype).transpose(1, 2).contiguous().reshape(batch, q_len, num_heads * head_dim)


def _split_dims(head_dim: int, outlier_channels: int) -> tuple[int, int]:
    assert 0 < outlier_channels < head_dim
    return head_dim - outlier_channels, outlier_channels


def _store_nbytes(runtime_config: RuntimeConfig, tc, max_len: int, dim: int, bits: int, kind: str) -> int:
    if runtime_config.turboquant_mode == "mse" or kind == "value":
        return tc.num_key_value_heads * max_len * (_packed_dim(dim, bits) * 4 + 2)
    elem_size = torch.empty(0, dtype=torch.bfloat16).element_size()
    return tc.num_key_value_heads * max_len * (_packed_dim(dim, bits - 1) * 4 + _packed_dim(dim, 1) * 4 + 2 * elem_size)
