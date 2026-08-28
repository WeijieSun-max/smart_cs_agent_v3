"""按用户和会话隔离 LangGraph checkpoint 线程。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver

# @dataclass(frozen=True) 这是 Python 的数据类写法。它会自动生成 __init__ 等方法，例如你可以这样创建对象
# identity = ThreadIdentity(user_id="user001", session_id="chat123") 表示这个对象不能再修改
# identity.user_id = "user002"  # 会报错
@dataclass(frozen=True)
class ThreadIdentity:
    """根据user_id + session_id
    生成一个唯一的线程ID，确保每个用户会话都有独立的检查点保存环境。

    这个 thread_id 会传给 LangGraph，用来区分不同用户、不同会话的工作流状态。
    """
    user_id: Optional[str]
    session_id: str

    @property
    def thread_id(self) -> str:
        """生成稳定的 `user_id:session_id` checkpoint 分区键。"""

        user = self.user_id or "anonymous"
        return f"{user}:{self.session_id}"

# LangGraph 的 checkpoint 是按线程保存状态的。
# LangGraph 可以通过同一个 threadid 恢复之前的状态。
class CheckpointSaverService:
    """封装 checkpointer，并统一线程键生成与可选清理能力。"""

    def __init__(self, checkpointer: BaseCheckpointSaver):
        self._checkpointer = checkpointer

    def get_checkpointer(self) -> BaseCheckpointSaver:
        """返回供工作流编译使用的底层保存器。"""

        return self._checkpointer

    @staticmethod
    def get_thread_id(user_id: Optional[str], session_id: str) -> str:
        """生成与 ThreadIdentity 一致的线程 ID。"""

        return ThreadIdentity(user_id=user_id, session_id=session_id).thread_id

    def clear_thread(self, user_id: Optional[str], session_id: str) -> None:
        """在保存器支持时删除指定用户会话的 checkpoint。"""

        delete_thread = getattr(self._checkpointer, "delete_thread", None)
        if callable(delete_thread):
            delete_thread(self.get_thread_id(user_id, session_id))

# 内存 checkpoint，所以服务重启后 checkpoint 会丢失
instance: Optional[CheckpointSaverService] = None


def initialize_service(checkpointer: BaseCheckpointSaver) -> None:
    """安装基础设施层提供的进程级 checkpoint 保存器。"""

    global instance
    instance = CheckpointSaverService(checkpointer=checkpointer)
