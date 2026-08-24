from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IConversationArchive(ABC):
    @property
    @abstractmethod
    def available(self) -> bool:
        pass

    @abstractmethod
    def create_session(self, session_id: str, title: str, agent_id: str) -> dict[str, object] | None:
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> dict[str, object] | None:
        pass

    @abstractmethod
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: str,
        turn_id: str | None = None,
    ) -> bool:
        pass

    def get_message_by_turn(self, turn_id: str, role: str) -> dict[str, str] | None:
        return None

    def complete_turn(
        self,
        session_id: str,
        content: str,
        timestamp: str,
        turn_id: str,
        *,
        enqueue_memory: bool = False,
    ) -> bool:
        del enqueue_memory
        return self.add_message(session_id, "assistant", content, timestamp, turn_id=turn_id)

    @abstractmethod
    def get_history(self, session_id: str, last_n: int) -> list[dict[str, str]]:
        pass

    @abstractmethod
    def list_sessions(self) -> list[dict[str, object]]:
        pass

    @abstractmethod
    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        favorite: bool | None = None,
    ) -> dict[str, object] | None:
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        pass

    @abstractmethod
    def start_run(self, session_id: str, turn_id: str, started_at: str) -> None:
        pass

    @abstractmethod
    def mark_run_stop_requested(self, turn_id: str) -> None:
        pass

    @abstractmethod
    def finish_run(self, turn_id: str, status: str, completed_at: str) -> None:
        pass

    @abstractmethod
    def record_run_step(
        self,
        turn_id: str,
        sequence_no: int,
        step_type: str,
        status: str,
        title: str,
        payload: dict[str, Any],
        created_at: str,
    ) -> None:
        pass

    @abstractmethod
    def list_runs(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        pass
