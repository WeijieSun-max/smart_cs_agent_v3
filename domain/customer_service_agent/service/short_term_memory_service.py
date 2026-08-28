"""近期会话端口的薄领域门面与进程级访问入口。"""

from __future__ import annotations

from typing import Optional

from domain.customer_service_agent.interfaces.i_short_term_memory import IShortTermMemory


class ShortTermMemoryService:
    """保持应用层与具体 Redis/持久化组合实现解耦的转发服务。"""

    def __init__(self, memory: IShortTermMemory):
        self.memory = memory

    def add_message(self, session_id: str, role: str, content: str, turn_id: str | None = None) -> None:
        """追加近期消息。"""

        self.memory.add_message(session_id, role, content, turn_id=turn_id)

    def complete_turn(
        self,
        session_id: str,
        content: str,
        turn_id: str,
        *,
        enqueue_memory: bool = False,
    ) -> None:
        """完成助手轮次，并按配置请求创建长期记忆事件。"""

        self.memory.complete_turn(
            session_id,
            content,
            turn_id,
            enqueue_memory=enqueue_memory,
        )

    def get_message_by_turn(self, turn_id: str, role: str) -> dict[str, str] | None:
        """读取用于幂等重放的轮次消息。"""

        return self.memory.get_message_by_turn(turn_id, role)

    def get_history(self, session_id: str, last_n: int | None = None) -> list[dict[str, str]]:
        """返回会话近期历史。"""

        return self.memory.get_history(session_id, last_n=last_n)

    def get_context_window(self, session_id: str, max_chars: int = 4000) -> str:
        """返回字符预算内的纯文本上下文。"""

        return self.memory.get_context_window(session_id, max_chars=max_chars)

    def create_session(
        self,
        session_id: str,
        title: str = "新会话",
        agent_id: str = "general",
    ) -> dict[str, object]:
        """创建或取得会话。"""

        return self.memory.create_session(session_id, title=title, agent_id=agent_id)

    def list_sessions(self) -> list[dict[str, object]]:
        """列出当前作用域会话。"""

        return self.memory.list_sessions()

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        favorite: bool | None = None,
    ) -> dict[str, object] | None:
        """更新会话标题或收藏状态。"""

        return self.memory.update_session(session_id, title=title, favorite=favorite)

    def delete_session(self, session_id: str) -> bool:
        """删除会话热状态。"""

        return self.memory.delete_session(session_id)


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
