import json
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime
from functools import wraps
from inspect import iscoroutinefunction
from pathlib import Path
from time import perf_counter
from typing import ParamSpec, TypeVar

import torch

P = ParamSpec("P")
R = TypeVar("R")


def format_console_line(prefix: str, message: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    level = prefix.strip().strip("[]").upper()
    return f"{timestamp} | {level} | {message}"


def read_events(path: str | Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


class Tracer:
    def __init__(self) -> None:
        self._file = None
        self._path: Path | None = None
        self._start_time = perf_counter()

    def enable(self, path: str | None = None) -> None:
        if path is None:
            path = os.getenv("ROVER_TRACE_FILE", "rover_trace.jsonl")
        trace_path = Path(path).expanduser()
        if self._file is not None and self._path == trace_path:
            return
        if self._file is not None:
            self._file.close()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = trace_path.open("w", encoding="utf-8", buffering=1)
        self._path = trace_path
        self._start_time = perf_counter()

    def flush(self) -> None:
        if self._file is not None:
            print(f"\n--- perf log written to {self._path} ---")
            self._file.flush()

    def elapsed_ms(self) -> float:
        return round((perf_counter() - self._start_time) * 1000.0, 3)

    def event(self, name: str, **fields) -> None:
        if self._file is None:
            return
        record = {
            "name": name,
            **fields,
            "elapsed_ms": self.elapsed_ms(),
        }
        self._file.write(json.dumps(record, ensure_ascii=True) + "\n")

    def timer(
        self,
        name: str,
        *,
        print_on_exit: bool = False,
        print_name: str | None = None,
        end_name: str | None = None,
        sync_cuda: bool = False,
        **fields: object,
    ) -> "Timer":
        return Timer(
            name,
            print_on_exit=print_on_exit,
            print_name=print_name,
            end_name=end_name,
            sync_cuda=sync_cuda,
            _backend=self,
            **fields,
        )

    def timed(
        self,
        name: str | None = None,
        *,
        print_on_exit: bool = False,
        print_name: str | None = None,
        end_name: str | None = None,
        fields_fn: Callable[P, dict[str, object]] | None = None,
        sync_cuda: bool = False,
        **fields: object,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            trace_name = func.__name__ if name is None else name
            trace_end_name = f"{trace_name}_end" if end_name is None else end_name

            if iscoroutinefunction(func):

                @wraps(func)
                async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                    dynamic_fields = {} if fields_fn is None else fields_fn(*args, **kwargs)
                    with self.timer(
                        trace_name,
                        print_on_exit=print_on_exit,
                        print_name=print_name,
                        end_name=trace_end_name,
                        sync_cuda=sync_cuda,
                        **fields,
                        **dynamic_fields,
                    ):
                        return await func(*args, **kwargs)

                return async_wrapper

            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                dynamic_fields = {} if fields_fn is None else fields_fn(*args, **kwargs)
                with self.timer(
                    trace_name,
                    print_on_exit=print_on_exit,
                    print_name=print_name,
                    end_name=trace_end_name,
                    sync_cuda=sync_cuda,
                    **fields,
                    **dynamic_fields,
                ):
                    return func(*args, **kwargs)

            return wrapper

        return decorator


class Timer(AbstractContextManager["Timer"]):
    def __init__(
        self,
        name: str,
        *,
        print_on_exit: bool = False,
        print_name: str | None = None,
        end_name: str | None = None,
        sync_cuda: bool = False,
        _backend: Tracer | None = None,
        **fields: object,
    ) -> None:
        self.name = name
        self.end_name = name if end_name is None else end_name
        self.fields = dict(fields)
        self.print_on_exit = print_on_exit
        self.print_name = print_name
        self.sync_cuda = sync_cuda
        self._backend = tracer if _backend is None else _backend
        self._started = False
        self._start_elapsed_ms = 0.0

    def start(self) -> "Timer":
        assert not self._started
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._started = True
        self._start_elapsed_ms = self._backend.elapsed_ms()
        self._backend.event(f"{self.name}_start", **self.fields)
        return self

    def elapsed_ms(self) -> float:
        return self._backend.elapsed_ms() - self._start_elapsed_ms

    def end(self, **fields) -> None:
        assert self._started
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = self._backend.elapsed_ms() - self._start_elapsed_ms
        self._backend.event(
            self.end_name,
            **self.fields,
            **fields,
            duration_ms=round(elapsed_ms, 3),
        )
        if self.print_on_exit:
            label = self.print_name if self.print_name is not None else self.name
            print(format_console_line("[timer]", f"{label}: {elapsed_ms:.3f} ms"))
        self._started = False

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end()


tracer = Tracer()
