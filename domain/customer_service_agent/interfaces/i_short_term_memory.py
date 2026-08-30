"""近期会话缓存的领域端口和兼容性默认行为。"""

from __future__ import annotations

from abc import ABC, abstractmethod


class IShortTermMemory(ABC):
    """定义消息热缓存与会话元数据操作；权威数据仍由归档持有。"""

    @abstractmethod
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        turn_id: str | None = None,
        *,
        user_id: str | None = None,
    ) -> None:
        """向会话追加近期消息。"""
        pass

    def complete_turn(
        self,
        session_id: str,
        content: str,
        turn_id: str,
        *,
        enqueue_memory: bool = False,
        user_id: str | None = None,
        fencing_token: int | None = None,
        lease_owner_id: str | None = None,
    ) -> None:
        """完成助手轮次；默认实现供不支持记忆 outbox 的缓存使用。"""
        del enqueue_memory, fencing_token, lease_owner_id
        if user_id is None:
            self.add_message(session_id, "assistant", content, turn_id=turn_id)
        else:
            self.add_message(session_id, "assistant", content, turn_id=turn_id, user_id=user_id)

    def get_session_summary(self, session_id: str, *, user_id: str | None = None) -> dict[str, object] | None:
        """读取可选摘要缓存；未实现时返回 None。"""
        del session_id, user_id
        return None

    def cache_session_summary(
        self,
        session_id: str,
        summary: dict[str, object],
        *,
        user_id: str | None = None,
    ) -> None:
        """写入摘要缓存；默认实现为空操作。"""
        del session_id, summary, user_id

    def delete_session_summary(self, session_id: str, *, user_id: str | None = None) -> None:
        """使摘要缓存失效；默认实现为空操作。"""
        del session_id, user_id

    @abstractmethod
    def get_message_by_turn(
        self,
        turn_id: str,
        role: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, str] | None:
        """按 turn_id 查找消息，用于请求重放与幂等响应。"""
        pass

    @abstractmethod
    def get_history(
        self,
        session_id: str,
        last_n: int | None = None,
        *,
        user_id: str | None = None,
    ) -> list[dict[str, str]]:
        """按时间顺序返回全部或最近 last_n 条热消息。"""
        pass

    @abstractmethod
    def get_context_window(
        self,
        session_id: str,
        max_chars: int = 4000,
        *,
        user_id: str | None = None,
    ) -> str:
        """在字符预算内拼装供模型阅读的近期上下文。"""
        pass

    @abstractmethod
    def create_session(
        self,
        session_id: str,
        title: str = "新会话",
        agent_id: str = "general",
        *,
        user_id: str | None = None,
    ) -> dict[str, object]:
        """创建或幂等取得会话元数据。"""
        pass

    @abstractmethod
    def list_sessions(self, *, user_id: str | None = None) -> list[dict[str, object]]:
        """列出当前用户的会话缓存摘要。"""
        pass

    @abstractmethod
    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        favorite: bool | None = None,
        user_id: str | None = None,
    ) -> dict[str, object] | None:
        """部分更新会话展示信息。"""
        pass

    @abstractmethod
    def delete_session(self, session_id: str, *, user_id: str | None = None) -> bool:
        """清除会话热数据并返回是否存在。"""
        pass
