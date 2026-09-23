from __future__ import annotations

import threading
from datetime import datetime, timezone
from uuid import uuid4

from pkg.log.logger import get_logger
from pkg.telemetry import normalize_error

logger = get_logger()


class ActionReconcileWorker:
    def __init__(self, store, poll_seconds: float = 5.0):
        self.store = store
        self.poll_seconds = poll_seconds
        self.worker_id = f"action-reconcile-{uuid4().hex[:10]}"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_success_at: datetime | None = None
        self._last_error_code: str | None = None

    def run_once(self) -> int:
        reconciled = 0
        for action in self.store.list_indeterminate(limit=20):
            receipt = self.store.get_receipt(action["idempotency_key"])
            if receipt is None:
                continue
            self.store.transition_pending(
                action["action_id"],
                action["user_id"],
                action["session_id"],
                "indeterminate",
                "succeeded",
                receipt_json=receipt,
                executed_at=datetime.now(timezone.utc),
            )
            reconciled += 1
        self._last_success_at = datetime.now(timezone.utc)
        self._last_error_code = None
        return reconciled

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name=self.worker_id,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def health_status(self) -> dict[str, object]:
        running = bool(self._thread is not None and self._thread.is_alive())
        return {
            "status": "ready" if running and self._last_error_code is None else "degraded" if self._last_error_code else "stopped",
            "running": running,
            "worker_id": self.worker_id,
            "last_success_at": self._last_success_at.isoformat() if self._last_success_at else None,
            "last_error_code": self._last_error_code,
        }

    def _run_safely(self) -> int:
        try:
            return self.run_once()
        except Exception as exc:
            error = normalize_error(exc)
            self._last_error_code = str(error["error_code"])
            logger.warning(
                "Action reconcile worker failed error_type={} error_code={}",
                error["error_type"],
                error["error_code"],
            )
            return 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._run_safely()
            self._stop.wait(self.poll_seconds)
