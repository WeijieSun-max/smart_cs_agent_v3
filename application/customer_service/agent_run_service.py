from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from domain.customer_service_agent.service import conversation_archive_service
from pkg.log.logger import get_logger
from pkg.config.settings import get_settings

logger = get_logger()
_PERSISTENCE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-run-persistence")

RunStatus = Literal["running", "completed", "failed", "stopped", "cancelled"]


@dataclass
class AgentRun:
    session_id: str
    turn_id: str
    user_id: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    status: RunStatus = "running"
    stop_requested: threading.Event = field(default_factory=threading.Event, repr=False)
    task: asyncio.Task | None = field(default=None, repr=False)
    loop: asyncio.AbstractEventLoop | None = field(default=None, repr=False)
    steps: list[dict[str, Any]] = field(default_factory=list)

    def attach_current_task(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.task = asyncio.current_task()

    def request_stop(self) -> None:
        self.stop_requested.set()
        if self.loop is not None and self.task is not None and not self.task.done():
            self.loop.call_soon_threadsafe(self.task.cancel)

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "status": self.status,
            "running": self.status == "running",
            "stop_requested": self.stop_requested.is_set(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "steps": list(self.steps),
        }


class AgentRunRegistry:
    def __init__(self) -> None:
        self._runs: dict[str, AgentRun] = {}
        self._lock = threading.RLock()
        self._ttl_seconds = get_settings().agent_state_ttl_seconds

    def begin(self, session_id: str, turn_id: str, *, user_id: str | None = None) -> AgentRun:
        with self._lock:
            self._cleanup_locked()
            current = self._runs.get(session_id)
            if current is not None and current.status == "running":
                raise RuntimeError(f"Session {session_id} already has a running agent")
            run = AgentRun(session_id=session_id, turn_id=turn_id, user_id=user_id)
            self._runs[session_id] = run
        try:
            def persist_start(archive) -> None:
                if user_id is None:
                    archive.start_run(session_id, turn_id, run.started_at)
                else:
                    archive.start_run(
                        session_id,
                        turn_id,
                        run.started_at,
                        user_id=user_id,
                    )

            self._persist(
                persist_start,
                strict=True,
            )
        except Exception:
            with self._lock:
                if self._runs.get(session_id) is run:
                    self._runs.pop(session_id, None)
            raise
        return run

    def get(self, session_id: str, *, user_id: str | None = None) -> AgentRun | None:
        with self._lock:
            self._cleanup_locked()
            run = self._runs.get(session_id)
            return run if run is not None and self._matches_user(run, user_id) else None

    def stop(self, session_id: str, *, user_id: str | None = None) -> AgentRun | None:
        with self._lock:
            run = self._runs.get(session_id)
            if run is None or not self._matches_user(run, user_id):
                return None
            if run.status == "running":
                run.request_stop()
                persist_turn_id = run.turn_id
            else:
                persist_turn_id = None
        if persist_turn_id is not None:
            self._persist(lambda archive: archive.mark_run_stop_requested(persist_turn_id))
        return run

    def finish(self, run: AgentRun, status: RunStatus) -> None:
        with self._lock:
            current = self._runs.get(run.session_id)
            if current is run:
                run.status = status
                run.task = None
                run.loop = None
                completed_at = datetime.now(timezone.utc).isoformat()
                run.completed_at = completed_at
            else:
                completed_at = None
        if completed_at is not None:
            self._persist(lambda archive: archive.finish_run(run.turn_id, status, completed_at))

    def start_step(
        self,
        run: AgentRun,
        title: str,
        *,
        node_name: str | None = None,
        step_type: str = "thinking",
        started_at: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            sequence_no = len(run.steps)
            started_at = started_at or datetime.now(timezone.utc).isoformat()
            step = {
                "id": f"{run.turn_id}-{sequence_no}",
                "type": step_type,
                "status": "running",
                "title": title,
                "toolName": node_name,
                "startedAt": started_at,
            }
            run.steps.append(step)
            result = dict(step)
        self._persist_step(run, sequence_no, step)
        return result

    def complete_step(
        self,
        run: AgentRun,
        step_id: str,
        *,
        title: str | None = None,
        status: str = "success",
        completed_at: str | None = None,
        duration_ms: int | None = None,
        token_usage: dict[str, int] | None = None,
        model_calls: int = 0,
    ) -> dict[str, Any]:
        with self._lock:
            sequence_no = next(
                (index for index, candidate in enumerate(run.steps) if candidate["id"] == step_id),
                -1,
            )
            if sequence_no < 0:
                raise KeyError(f"Unknown agent step: {step_id}")
            step = run.steps[sequence_no]
            completed_at = completed_at or datetime.now(timezone.utc).isoformat()
            step.update({"status": status, "completedAt": completed_at})
            if title:
                step["title"] = title
            if duration_ms is not None:
                step["duration"] = max(0, duration_ms)
            if model_calls:
                step["modelCalls"] = model_calls
            if token_usage is not None:
                step["tokenUsage"] = token_usage
            result = dict(step)
        self._persist_step(run, sequence_no, step)
        return result

    def record_step(
        self,
        run: AgentRun,
        title: str,
        *,
        node_name: str | None = None,
        step_type: str = "thinking",
        status: str = "success",
        started_at: str | None = None,
        completed_at: str | None = None,
        duration_ms: int | None = 0,
        token_usage: dict[str, int] | None = None,
        model_calls: int = 0,
    ) -> dict[str, Any]:
        step = self.start_step(
            run,
            title,
            node_name=node_name,
            step_type=step_type,
            started_at=started_at,
        )
        return self.complete_step(
            run,
            step["id"],
            status=status,
            completed_at=completed_at,
            duration_ms=duration_ms,
            token_usage=token_usage,
            model_calls=model_calls,
        )

    def _persist_step(self, run: AgentRun, sequence_no: int, step: dict[str, Any]) -> None:
        payload = {
            "node_name": step.get("toolName"),
            "description": step.get("description"),
            "input": step.get("input"),
            "output": step.get("output"),
            "completed_at": step.get("completedAt"),
            "duration_ms": step.get("duration"),
            "token_usage": step.get("tokenUsage"),
            "model_calls": step.get("modelCalls", 0),
        }
        self._persist(lambda archive: archive.record_run_step(
            run.turn_id,
            sequence_no,
            str(step["type"]),
            str(step["status"]),
            str(step["title"]),
            payload,
            str(step["startedAt"]),
        ))

    def list_runs(self, session_id: str, *, user_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            self._cleanup_locked()
            runs = [
                run.as_dict()
                for run in self._runs.values()
                if run.session_id == session_id and self._matches_user(run, user_id)
            ]
        return sorted(runs, key=lambda run: str(run["started_at"]), reverse=True)

    def forget(self, session_id: str, *, user_id: str | None = None) -> None:
        with self._lock:
            run = self._runs.get(session_id)
            if run is not None and self._matches_user(run, user_id):
                self._runs.pop(session_id, None)

    @staticmethod
    def _matches_user(run: AgentRun, user_id: str | None) -> bool:
        return user_id is None or run.user_id == user_id

    def _cleanup_locked(self) -> None:
        now = datetime.now(timezone.utc)
        expired: list[str] = []
        for session_id, run in self._runs.items():
            if run.status == "running" or not run.completed_at:
                continue
            try:
                completed = datetime.fromisoformat(run.completed_at.replace("Z", "+00:00"))
            except ValueError:
                expired.append(session_id)
                continue
            if (now - completed).total_seconds() > self._ttl_seconds:
                expired.append(session_id)
        for session_id in expired:
            self._runs.pop(session_id, None)

    @staticmethod
    def _persist(operation, *, strict: bool = False) -> None:
        def execute() -> None:
            archive = conversation_archive_service.get_service_or_none()
            if archive is None:
                return
            try:
                operation(archive)
            except Exception as exc:
                logger.warning("Agent run persistence failed error_type={}", type(exc).__name__)
                if strict:
                    raise

        if strict:
            # Queue strict run creation behind any completion writes from a
            # preceding attempt, then surface its result to the caller.
            _PERSISTENCE_EXECUTOR.submit(execute).result()
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            execute()
        else:
            # A single queue preserves start/complete/finish write ordering
            # while keeping synchronous MySQL calls off the event loop.
            loop.run_in_executor(_PERSISTENCE_EXECUTOR, execute)


registry = AgentRunRegistry()
