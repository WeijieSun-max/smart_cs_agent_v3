from __future__ import annotations

from abc import ABC, abstractmethod


class IShortTermMemory(ABC):
    @abstractmethod
    def add_message(self, session_id: str, role: str, content: str, turn_id: str | None = None) -> None:
        pass

    def complete_turn(
        self,
        session_id: str,
        content: str,
        turn_id: str,
        *,
        enqueue_memory: bool = False,
    ) -> None:
        del enqueue_memory
        self.add_message(session_id, "assistant", content, turn_id=turn_id)

    def get_session_summary(self, session_id: str) -> dict[str, object] | None:
        return None

    def cache_session_summary(self, session_id: str, summary: dict[str, object]) -> None:
        del session_id, summary

    def delete_session_summary(self, session_id: str) -> None:
        del session_id

    @abstractmethod
    def get_message_by_turn(self, turn_id: str, role: str) -> dict[str, str] | None:
        pass

    @abstractmethod
    def get_history(self, session_id: str, last_n: int | None = None) -> list[dict[str, str]]:
        pass

    @abstractmethod
    def get_context_window(self, session_id: str, max_chars: int = 4000) -> str:
        pass

    @abstractmethod
    def create_session(
        self,
        session_id: str,
        title: str = "新会话",
        agent_id: str = "general",
    ) -> dict[str, object]:
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
