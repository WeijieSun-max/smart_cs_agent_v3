"""长期记忆、摘要、来源审计和索引 outbox 的权威仓储端口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from domain.customer_service_agent.memory.models import (
    MemoryItem,
    MemoryOutboxEventType,
    MemorySource,
    MemoryStatus,
    MemoryType,
    SessionSummary,
)


class IMemoryRepository(ABC):
    """定义记忆事实源必须提供的一致性和生命周期操作。

    向量索引不是事实源：记忆新增、纠正、删除和 outbox 事件必须由仓储在同一
    事务内提交。所有读取都显式携带 `user_id`，实现不得先按 memory_id 查询
    后再在应用层判断所有权。
    """

    @property
    @abstractmethod
    def available(self) -> bool:
        """权威仓储当前是否可接受一致性读写。"""
        pass

    @abstractmethod
    def get_latest_summary(self, user_id: str, session_id: str) -> SessionSummary | None:
        """读取用户会话版本最高的摘要。"""
        pass

    @abstractmethod
    def get_messages_after(
        self,
        user_id: str,
        session_id: str,
        after_message_id: int,
        limit: int = 1000,
    ) -> list[dict[str, object]]:
        """读取摘要覆盖点之后的归档消息。"""
        pass

    @abstractmethod
    def get_items_by_ids(self, user_id: str, memory_ids: list[str], now: datetime) -> list[MemoryItem]:
        """按用户隔离批量读取当前仍有效的指定记忆。"""
        pass

    @abstractmethod
    def list_active_items(
        self,
        user_id: str,
        *,
        memory_types: list[MemoryType] | None = None,
        limit: int = 20,
        now: datetime,
    ) -> list[MemoryItem]:
        """列出用户活跃记忆，可按类型过滤并在数据库层限制数量。"""
        pass

    @abstractmethod
    def scan_active_items(
        self,
        *,
        cursor: str | None,
        limit: int,
        now: datetime,
    ) -> tuple[list[MemoryItem], str | None]:
        """游标扫描全部用户的活跃记忆，供索引重建使用。"""
        pass

    @abstractmethod
    def find_active_by_key(self, user_id: str, memory_type: MemoryType, memory_key: str) -> MemoryItem | None:
        """查找同用户、类型和规范化键的当前活跃版本。"""
        pass

    @abstractmethod
    def apply_extraction(
        self,
        summary: SessionSummary,
        items: list[MemoryItem],
        sources: list[MemorySource],
        superseded_ids: list[str],
    ) -> None:
        """原子保存摘要、提取记忆、来源关系、取代关系和索引事件。"""
        pass

    @abstractmethod
    def list_items(
        self,
        user_id: str,
        *,
        memory_type: MemoryType | None = None,
        status: MemoryStatus | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> tuple[list[MemoryItem], str | None]:
        """按用户、类型和状态分页列出记忆管理视图。"""
        pass

    @abstractmethod
    def correct_item(
        self,
        user_id: str,
        memory_id: str,
        *,
        content: str,
        structured_data: dict[str, Any] | None,
        reason: str,
        actor: str,
    ) -> MemoryItem | None:
        """创建人工纠正版本并审计旧值、新值、原因和操作者。"""
        pass

    @abstractmethod
    def hard_delete_item(self, user_id: str, memory_id: str, *, reason: str, actor: str) -> bool:
        """按所有权永久删除单条记忆及向量，并记录不可逆操作审计。"""
        pass

    @abstractmethod
    def purge_items(self, user_id: str, memory_type: MemoryType | None, *, reason: str, actor: str) -> int:
        """永久删除用户全部或指定类型记忆，返回删除数量。"""
        pass

    @abstractmethod
    def delete_session_sources(self, user_id: str, session_id: str) -> list[str]:
        """删除会话来源，并返回因失去全部来源而被同步删除的记忆 ID。"""
        pass

    @abstractmethod
    def claim_outbox(
        self,
        event_types: list[MemoryOutboxEventType],
        *,
        worker_id: str,
        limit: int,
        lease_seconds: int,
        now: datetime,
    ) -> list[MemoryOutboxEvent]:
        """以租约方式领取可处理 outbox 事件，避免多 worker 重复消费。"""
        pass

    @abstractmethod
    def complete_outbox(self, event_id: str, processed_at: datetime) -> None:
        """将成功处理的 outbox 事件标记为完成。"""
        pass

    @abstractmethod
    def retry_outbox(
        self,
        event_id: str,
        *,
        error_code: str,
        available_at: datetime,
        max_attempts: int,
    ) -> None:
        """记录失败并延迟重试；超过上限时实现应转入 dead-letter。"""
        pass

    @abstractmethod
    def outbox_stats(self, now: datetime) -> dict[str, int | float | None]:
        """返回积压量、死信数和最老事件延迟等运维指标。"""
        pass

    @abstractmethod
    def replay_dead_letter(self, event_id: str, now: datetime) -> bool:
        """把指定死信事件恢复为可消费状态。"""
        pass
