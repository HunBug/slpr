from __future__ import annotations

import threading
import time
from typing import Optional

from tqdm import tqdm


class ProgressCounter:
    """Lightweight counter that increments a shared integer across processes.

    Wraps a multiprocessing Manager.Value proxy provided by the main process.
    """

    def __init__(self, shared_value) -> None:  # ValueProxy[int]
        self._v = shared_value

    def add(self, n: int) -> None:
        if n <= 0:
            return
        try:
            lock = self._v.get_lock()  # type: ignore[attr-defined]
        except Exception:
            # Fallback for Manager.Value proxies that may not expose get_lock
            self._v.value = int(self._v.value) + int(n)
            return
        with lock:
            self._v.value += int(n)


class GlobalProgressBar:
    """Background thread that renders a tqdm bar for a shared counter."""

    def __init__(
        self,
        shared_value,
        total: int,
        desc: str = "Progress",
        interval: float = 0.25,
    ) -> None:
        self._v = shared_value
        self._total = int(total)
        self._desc = desc
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._bar: Optional[tqdm] = None
        self._interval = float(interval)

    def start(self) -> None:
        self._bar = tqdm(total=self._total, desc=self._desc, unit="px")

        def _run() -> None:
            assert self._bar is not None
            last_n = 0
            while not self._stop.is_set():
                try:
                    lock = self._v.get_lock()  # type: ignore[attr-defined]
                except Exception:
                    n = int(self._v.value)
                else:
                    with lock:
                        n = int(self._v.value)
                if n > last_n:
                    self._bar.update(n - last_n)
                    last_n = n
                time.sleep(self._interval)
            # Final sync
            try:
                lock = self._v.get_lock()  # type: ignore[attr-defined]
            except Exception:
                n = int(self._v.value)
            else:
                with lock:
                    n = int(self._v.value)
            self._bar.n = n
            self._bar.refresh()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._bar is not None:
            self._bar.close()
