from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from rover.models.spec import ModelShape
from rover.runtime.config import RuntimeConfig

if TYPE_CHECKING:
    from rover.runtime.types import Range
    from rover.storage.files import CacheFile


class BaseKVCache(ABC):
    @staticmethod
    @abstractmethod
    def compute_file_sizes(
        *,
        config: ModelShape,
        runtime_config: RuntimeConfig,
        max_len: int,
    ) -> tuple[int, int]: ...

    @staticmethod
    @abstractmethod
    def build(
        *,
        config: ModelShape,
        runtime_config: RuntimeConfig,
        max_len: int,
        cache_metadata: dict | None = None,
    ) -> BaseKVCache: ...

    @abstractmethod
    def cache_format(self) -> str: ...

    @abstractmethod
    def update_kv(self, r: Range, key_states: torch.Tensor, value_states: torch.Tensor) -> None: ...

    @abstractmethod
    def save_prefill_layer(self, layer: int, start: int, end: int, k_file: CacheFile, v_file: CacheFile) -> None: ...

    @abstractmethod
    def load_prefill_layer(self, layer: int, start: int, end: int, k_file: CacheFile, v_file: CacheFile) -> None: ...

    @abstractmethod
    def prefill_key_file_size(self) -> int: ...

    @abstractmethod
    def prefill_value_file_size(self) -> int: ...

    @abstractmethod
    def begin_decode(self, prefix_len: int) -> None: ...

    @abstractmethod
    def decode_record_nbytes(self) -> int: ...

    @abstractmethod
    def save_decode_tokens(self, start_token: int, end_token: int, decode_file: CacheFile) -> None: ...

    @abstractmethod
    def load_decode_tokens(self, decode_file: CacheFile, num_tokens: int) -> None: ...

    @abstractmethod
    def get_metadata(self) -> dict: ...

    @abstractmethod
    def get_kv_for_attention(self, layer: int): ...

    def metadata_payload(self, base_dir: Path) -> dict:
        return self.get_metadata()

    @classmethod
    def resolve_metadata_payload(cls, base_dir: Path, payload: dict) -> dict:
        return payload


def _kv_cache_cls(cache_format: str) -> type[BaseKVCache]:
    if cache_format == "dense":
        from rover.cache.dense import DenseKVCache

        return DenseKVCache
    if cache_format == "kivi":
        from rover.cache.kivi import KIVIKVCache

        return KIVIKVCache
    if cache_format == "turboquant":
        from rover.cache.turboquant import TurboQuantKVCache

        return TurboQuantKVCache
    raise ValueError(f"unknown kv cache format: {cache_format}")


def compute_kv_file_sizes(
    *,
    config: ModelShape,
    runtime_config: RuntimeConfig,
    max_len: int,
    cache_format: str,
) -> tuple[int, int]:
    return _kv_cache_cls(cache_format).compute_file_sizes(
        config=config,
        runtime_config=runtime_config,
        max_len=max_len,
    )


def build_kv_cache(
    *,
    config: ModelShape,
    runtime_config: RuntimeConfig,
    max_len: int,
    cache_format: str,
    cache_metadata: dict | None = None,
) -> BaseKVCache:
    return _kv_cache_cls(cache_format).build(
        config=config,
        runtime_config=runtime_config,
        max_len=max_len,
        cache_metadata=cache_metadata,
    )


def resolve_kv_cache_metadata(*, cache_format: str, base_dir: Path, cache_metadata: dict) -> dict:
    return _kv_cache_cls(cache_format).resolve_metadata_payload(base_dir, cache_metadata)
