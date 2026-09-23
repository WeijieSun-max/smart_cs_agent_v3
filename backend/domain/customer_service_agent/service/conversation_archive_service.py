"""权威会话归档端口的进程级初始化与访问入口。"""

from __future__ import annotations

from typing import Optional

from domain.customer_service_agent.interfaces.i_conversation_archive import IConversationArchive

instance: Optional[IConversationArchive] = None


def initialize_service(archive: IConversationArchive) -> None:
    """安装基础设施层提供的会话归档实现。"""

    global instance
    instance = archive


def get_service() -> IConversationArchive:
    """返回归档；启动顺序错误时立即失败。"""

    if instance is None:
        raise RuntimeError("Conversation archive service is not initialized")
    return instance


def get_service_or_none() -> IConversationArchive | None:
    """供允许降级的调用方探测归档是否已经安装。"""

    return instance
