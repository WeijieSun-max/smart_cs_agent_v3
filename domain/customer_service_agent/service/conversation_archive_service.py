from __future__ import annotations

from typing import Optional

from domain.customer_service_agent.interfaces.i_conversation_archive import IConversationArchive

instance: Optional[IConversationArchive] = None


def initialize_service(archive: IConversationArchive) -> None:
    global instance
    instance = archive


def get_service() -> IConversationArchive:
    if instance is None:
        raise RuntimeError("Conversation archive service is not initialized")
    return instance


def get_service_or_none() -> IConversationArchive | None:
    return instance
