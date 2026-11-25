from dataclasses import asdict, dataclass
from datetime import datetime

import torch


@dataclass(frozen=True)
class GpuMemStats:
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    max_reserved_mb: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def capture_gpu_mem() -> GpuMemStats | None:
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return GpuMemStats(
        allocated_mb=torch.cuda.memory_allocated() / 1024**2,
        reserved_mb=torch.cuda.memory_reserved() / 1024**2,
        max_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2,
        max_reserved_mb=torch.cuda.max_memory_reserved() / 1024**2,
    )


def gpu_mem(tag=""):
    stats = capture_gpu_mem()
    if stats is None:
        print(format_console_line(f"[{tag}]", "CUDA not available"))
        return
    print(
        format_console_line(
            f"[{tag}]",
            f"allocated={stats.allocated_mb:.1f}MB "
            f"reserved={stats.reserved_mb:.1f}MB "
            f"max_allocated={stats.max_allocated_mb:.1f}MB "
            f"max_reserved={stats.max_reserved_mb:.1f}MB",
        )
    )


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def format_console_line(prefix: str, message: str) -> str:
    return f"{datetime.now().strftime('%H:%M:%S')} {prefix} {message}"
