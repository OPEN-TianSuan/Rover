from __future__ import annotations

import threading
from concurrent.futures import Future
from queue import Queue
from typing import Callable, TypeVar

T = TypeVar("T")

_SENTINEL = None


class IOThread:
    def __init__(self):
        self._queue: Queue = Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="rover-io")
        self._thread.start()

    def _run(self):
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                return
            fn, future = item
            try:
                future.set_result(fn())
            except Exception as e:
                future.set_exception(e)

    def submit_async(self, fn: Callable[[], T]) -> Future[T]:
        future: Future[T] = Future()
        self._queue.put((fn, future))
        return future

    def submit_sync(self, fn: Callable[[], T]) -> T:
        return self.submit_async(fn).result()

    def shutdown(self):
        self._queue.put(_SENTINEL)
        self._thread.join()


_global_io_thread: IOThread | None = None
_lock = threading.Lock()


def io_thread() -> IOThread:
    global _global_io_thread
    if _global_io_thread is None:
        with _lock:
            if _global_io_thread is None:
                _global_io_thread = IOThread()
    return _global_io_thread
