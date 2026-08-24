from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any


class CircuitBreaker:
    def __init__(self,failure_threshold: int=5,recovery_seconds: float=30.0):
        self.failure_threshold=failure_threshold; self.recovery_seconds=recovery_seconds
        self._failures=0; self._opened_at: float|None=None; self._lock=threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None: return "closed"
            return "half_open" if time.monotonic()-self._opened_at>=self.recovery_seconds else "open"

    def call(self,operation: Callable[...,Any],*args: Any,**kwargs: Any) -> Any:
        with self._lock:
            if self._opened_at is not None and time.monotonic()-self._opened_at<self.recovery_seconds:
                raise RuntimeError("circuit breaker is open")
        try: result=operation(*args,**kwargs)
        except Exception:
            with self._lock:
                self._failures+=1
                if self._failures>=self.failure_threshold: self._opened_at=time.monotonic()
            raise
        with self._lock: self._failures=0; self._opened_at=None
        return result
