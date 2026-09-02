"""近期会话端口的薄领域门面与进程级访问入口。"""

from __future__ import annotations

from typing import Optional

from domain.customer_service_agent.interfaces.i_short_term_memory import IShortTermMemory


class ShortTermMemoryService:
    """保持应用层与具体 Redis/持久化组合实现解耦的转发服务。"""

    def __init__(self, memory: IShortTermMemory):
        self.memory = memory

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        turn_id: str | None = None,
        *,
        user_id: str | None = None,
    ) -> None:
        """追加近期消息。"""

        if user_id is None:
            self.memory.add_message(session_id, role, content, turn_id=turn_id)
        else:
            self.memory.add_message(session_id, role, content, turn_id=turn_id, user_id=user_id)

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
        """完成助手轮次，并按配置请求创建长期记忆事件。"""

        kwargs = {"enqueue_memory": enqueue_memory}
        if user_id is not None:
            kwargs["user_id"] = user_id
        if fencing_token is not None:
            kwargs["fencing_token"] = fencing_token
            kwargs["lease_owner_id"] = lease_owner_id
        self.memory.complete_turn(session_id, content, turn_id, **kwargs)

    def get_message_by_turn(
        self,
        turn_id: str,
        role: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, str] | None:
        """读取用于幂等重放的轮次消息。"""

        if user_id is None:
            return self.memory.get_message_by_turn(turn_id, role)
        return self.memory.get_message_by_turn(turn_id, role, user_id=user_id)

    def get_history(
        self,
        session_id: str,
        last_n: int | None = None,
        *,
        user_id: str | None = None,
    ) -> list[dict[str, str]]:
        """返回会话近期历史。"""

        if user_id is None:
            return self.memory.get_history(session_id, last_n=last_n)
        return self.memory.get_history(session_id, last_n=last_n, user_id=user_id)

    def get_context_window(
        self,
        session_id: str,
        max_chars: int = 4000,
        *,
        user_id: str | None = None,
    ) -> str:
        """返回字符预算内的纯文本上下文。"""

        if user_id is None:
            return self.memory.get_context_window(session_id, max_chars=max_chars)
        return self.memory.get_context_window(session_id, max_chars=max_chars, user_id=user_id)

    def get_pending_task(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, object] | None:
        """读取当前会话等待续接的结构化任务。"""

        return self.memory.get_pending_task(session_id, user_id=user_id)

    def cache_pending_task(
        self,
        session_id: str,
        task: dict[str, object],
        *,
        user_id: str | None = None,
    ) -> None:
        """缓存当前会话等待续接的结构化任务。"""

        self.memory.cache_pending_task(session_id, task, user_id=user_id)

    def delete_pending_task(self, session_id: str, *, user_id: str | None = None) -> None:
        """清理已完成或已转为治理提案的待续接任务。"""

        self.memory.delete_pending_task(session_id, user_id=user_id)

    def create_session(
        self,
        session_id: str,
        title: str = "新会话",
        agent_id: str = "general",
        *,
        user_id: str | None = None,
    ) -> dict[str, object]:
        """创建或取得会话。"""

        if user_id is None:
            return self.memory.create_session(session_id, title=title, agent_id=agent_id)
        return self.memory.create_session(session_id, title=title, agent_id=agent_id, user_id=user_id)

    def list_sessions(self, *, user_id: str | None = None) -> list[dict[str, object]]:
        """列出当前作用域会话。"""

        if user_id is None:
            return self.memory.list_sessions()
        return self.memory.list_sessions(user_id=user_id)

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        favorite: bool | None = None,
        user_id: str | None = None,
    ) -> dict[str, object] | None:
        """更新会话标题或收藏状态。"""

        kwargs = {"title": title, "favorite": favorite}
        if user_id is not None:
            kwargs["user_id"] = user_id
        return self.memory.update_session(session_id, **kwargs)

    def delete_session(self, session_id: str, *, user_id: str | None = None) -> bool:
        """删除会话热状态。"""

        if user_id is None:
            return self.memory.delete_session(session_id)
        return self.memory.delete_session(session_id, user_id=user_id)


instance: Optional[ShortTermMemoryService] = None


def initialize_service(memory: IShortTermMemory) -> None:
    """安装具体近期记忆实现。"""

    global instance
    instance = ShortTermMemoryService(memory)


def get_service() -> ShortTermMemoryService:
    """返回已初始化的近期记忆服务。"""

    if instance is None:
        raise RuntimeError("Short-term memory service is not initialized")
    return instance
