"""Binary tensor storage and checkpoint file operations."""

import os
from pathlib import Path

import torch

from rover.storage.worker import io_thread


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def create_cache_file(path: Path, nbytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, nbytes)
    finally:
        os.close(fd)
    _fsync_dir(path.parent)


def swap_cache_file(path: Path, nbytes: int) -> tuple[Path, "CacheFile"]:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    create_cache_file(tmp_path, nbytes)
    return tmp_path, CacheFile(tmp_path)


class CacheFile:
    _IO_CHUNK_BYTES = 16 * 1024 * 1024

    def __init__(self, path: Path):
        self._fd = os.open(str(path), os.O_RDWR)

    def __enter__(self) -> "CacheFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _discard_page_cache(self, offset: int, length: int) -> None:
        if length <= 0 or not hasattr(os, "posix_fadvise"):
            return
        try:
            os.posix_fadvise(self._fd, offset, length, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass

    def sync(self) -> None:
        io_thread().submit_sync(lambda: os.fdatasync(self._fd))

    def discard_all_page_cache(self) -> None:
        self._discard_page_cache(0, os.fstat(self._fd).st_size)

    def write(self, data: bytes, offset: int):
        io_thread().submit_sync(lambda: os.pwrite(self._fd, data, offset))
        self._discard_page_cache(offset, len(data))

    def writev(self, buffers: list[bytes], offset: int):
        io_thread().submit_sync(lambda: os.pwritev(self._fd, buffers, offset))
        self._discard_page_cache(offset, sum(len(buf) for buf in buffers))

    def read(self, nbytes: int, offset: int) -> bytes:
        data = io_thread().submit_sync(lambda: os.pread(self._fd, nbytes, offset))
        self._discard_page_cache(offset, nbytes)
        return data

    def write_tensor(self, tensor: torch.Tensor, offset: int):
        tensor_cpu = tensor.contiguous().cpu()
        data = memoryview(tensor_cpu.view(torch.uint8).numpy()).cast("B")
        for start in range(0, tensor_cpu.nbytes, self._IO_CHUNK_BYTES):
            end = min(start + self._IO_CHUNK_BYTES, tensor_cpu.nbytes)
            self.write(data[start:end].tobytes(), offset + start)

    def read_tensor(self, tensor: torch.Tensor, nbytes: int, offset: int):
        src_buffer = torch.frombuffer(bytearray(self.read(nbytes, offset)), dtype=tensor.dtype).reshape(tensor.shape)
        tensor.copy_(src_buffer)

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def _save_tensor_checkpoint(path: Path, tensor: torch.Tensor) -> dict:
    tensor_cpu = tensor.contiguous().cpu()
    create_cache_file(path, tensor_cpu.nbytes)
    with CacheFile(path) as cache:
        cache.write_tensor(tensor_cpu, 0)
        cache.sync()
    return {
        "shape": list(tensor_cpu.shape),
        "dtype": str(tensor_cpu.dtype).replace("torch.", ""),
        "file": path.name,
    }


def _load_tensor_checkpoint(base: Path, meta: dict) -> torch.Tensor:
    tensor = torch.empty(meta["shape"], dtype=getattr(torch, meta["dtype"]))
    with CacheFile(base / meta["file"]) as cache:
        cache.read_tensor(tensor, tensor.nbytes, 0)
    return tensor
