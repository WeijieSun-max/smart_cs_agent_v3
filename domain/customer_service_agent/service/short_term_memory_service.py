from __future__ import annotations

from typing import Optional

from domain.customer_service_agent.interfaces.i_short_term_memory import IShortTermMemory


class ShortTermMemoryService:
    def __init__(self, memory: IShortTermMemory):
        self.memory = memory

    def add_message(self, session_id: str, role: str, content: str, turn_id: str | None = None) -> None:
        self.memory.add_message(session_id, role, content, turn_id=turn_id)

    def complete_turn(
        self,
        session_id: str,
        content: str,
        turn_id: str,
        *,
        enqueue_memory: bool = False,
    ) -> None:
        self.memory.complete_turn(
            session_id,
            content,
            turn_id,
            enqueue_memory=enqueue_memory,
        )

    def get_message_by_turn(self, turn_id: str, role: str) -> dict[str, str] | None:
        return self.memory.get_message_by_turn(turn_id, role)

    def get_history(self, session_id: str, last_n: int | None = None) -> list[dict[str, str]]:
        return self.memory.get_history(session_id, last_n=last_n)

    def get_context_window(self, session_id: str, max_chars: int = 4000) -> str:
        return self.memory.get_context_window(session_id, max_chars=max_chars)

    def create_session(
        self,
        session_id: str,
        title: str = "新会话",
        agent_id: str = "general",
    ) -> dict[str, object]:
        return self.memory.create_session(session_id, title=title, agent_id=agent_id)

    def list_sessions(self) -> list[dict[str, object]]:
        return self.memory.list_sessions()

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        favorite: bool | None = None,
    ) -> dict[str, object] | None:
        return self.memory.update_session(session_id, title=title, favorite=favorite)

    def delete_session(self, session_id: str) -> bool:
        return self.memory.delete_session(session_id)


instance: Optional[ShortTermMemoryService] = None


def initialize_service(memory: IShortTermMemory) -> None:
    global instance
    instance = ShortTermMemoryService(memory)


def get_service() -> ShortTermMemoryService:
    if instance is None:
        raise RuntimeError("Short-term memory service is not initialized")
    return instance
