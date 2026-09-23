"""权威会话归档及 Agent 运行轨迹的持久化端口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IConversationArchive(ABC):
    """定义会话、消息和运行步骤的长期存储契约。"""

    @property
    @abstractmethod
    def available(self) -> bool:
        """权威归档当前是否可接受一致性读写。"""
        pass

    @abstractmethod
    def create_session(
        self,
        session_id: str,
        title: str,
        agent_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, object] | None:
        """创建会话并返回持久化记录；ID 冲突时由实现安全处理。"""
        pass

    @abstractmethod
    def get_session(self, session_id: str, *, user_id: str | None = None) -> dict[str, object] | None:
        """按会话 ID 读取会话元数据。"""
        pass

    @abstractmethod
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: str,
        turn_id: str | None = None,
        *,
        user_id: str | None = None,
    ) -> bool:
        """追加一条带可选 turn_id 的消息。"""
        pass

    def get_message_by_turn(
        self,
        turn_id: str,
        role: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, str] | None:
        """按幂等 turn_id 查找指定角色消息；旧实现可返回 None。"""
        del turn_id, role, user_id
        return None

    def complete_turn(
        self,
        session_id: str,
        content: str,
        timestamp: str,
        turn_id: str,
        *,
        enqueue_memory: bool = False,
        user_id: str | None = None,
        fencing_token: int | None = None,
        lease_owner_id: str | None = None,
    ) -> bool:
        """原子完成助手轮次；默认实现仅兼容不支持 outbox 的归档。"""
        del enqueue_memory, fencing_token, lease_owner_id
        if user_id is None:
            return self.add_message(session_id, "assistant", content, timestamp, turn_id=turn_id)
        return self.add_message(
            session_id,
            "assistant",
            content,
            timestamp,
            turn_id=turn_id,
            user_id=user_id,
        )

    @abstractmethod
    def get_history(
        self,
        session_id: str,
        last_n: int,
        *,
        user_id: str | None = None,
    ) -> list[dict[str, str]]:
        """按时间顺序返回会话最近消息。"""
        pass

    @abstractmethod
    def list_sessions(self, *, user_id: str | None = None) -> list[dict[str, object]]:
        """列出调用方作用域内的会话摘要。"""
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
        """部分更新会话标题或收藏状态。"""
        pass

    @abstractmethod
    def delete_session(self, session_id: str, *, user_id: str | None = None) -> bool:
        """删除会话及其归档数据，返回是否实际删除。"""
        pass

    @abstractmethod
    def start_run(
        self,
        session_id: str,
        turn_id: str,
        started_at: str,
        *,
        user_id: str | None = None,
    ) -> None:
        """记录一个 Agent 轮次开始。"""
        pass

    @abstractmethod
    def mark_run_stop_requested(self, turn_id: str) -> None:
        """持久化协作式停止请求。"""
        pass

    @abstractmethod
    def finish_run(self, turn_id: str, status: str, completed_at: str) -> None:
        """记录轮次终态和完成时间。"""
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
        """按递增序号追加可审计的运行步骤。"""
        pass

    @abstractmethod
    def list_runs(
        self,
        session_id: str,
        limit: int = 50,
        *,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """返回会话最近运行及其步骤。"""
        pass
