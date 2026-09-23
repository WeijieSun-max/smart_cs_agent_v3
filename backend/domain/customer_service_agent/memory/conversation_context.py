"""把内部记忆包投影为 LLM 可见但不可信的最小会话上下文。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from pydantic import ValidationError

from domain.customer_service_agent.memory.models import (
    ConversationContext,
    ConversationContextMemory,
    ConversationMemoryMessage,
    MemoryPacket,
    MemoryType,
)


def build_conversation_context(packet: MemoryPacket) -> ConversationContext:
    """把已验证记忆包投影成更小的 LLM 契约，并移除存储标识。"""

    references = [*packet.episodes, *packet.semantic_memories]
    return ConversationContext(
        summary=packet.session_summary,
        recent_messages=packet.recent_messages,
        memories=[
            ConversationContextMemory(
                memory_type=reference.memory_type,
                content=reference.content,
                confidence=reference.confidence,
                score=reference.score,
                updated_at=reference.updated_at,
            )
            for reference in references
        ],
    )


def build_recent_conversation_context(
    messages: Sequence[ConversationMemoryMessage],
) -> ConversationContext:
    """在长期记忆关闭或不可用时，仅用近期消息构造上下文。"""

    return ConversationContext(recent_messages=list(messages))


def normalize_conversation_context(value: Any) -> ConversationContext:
    """规范化图状态中的上下文；畸形或旧结构按空上下文关闭处理。"""

    if isinstance(value, ConversationContext):
        return value
    if not isinstance(value, Mapping):
        return ConversationContext()
    try:
        return ConversationContext.model_validate(value)
    except (TypeError, ValueError, ValidationError):
        return ConversationContext()


def conversation_context_payload(value: Any) -> dict[str, Any]:
    """生成可 JSON 序列化、可安全放入模型提示的上下文。"""

    return normalize_conversation_context(value).model_dump(mode="json")


def task_scoped_context_payload(value: Any) -> dict[str, Any]:
    """投影给领域子 Agent 的最小上下文。

    当前消息和记忆尚未携带可靠领域标签，因此不把全局摘要、历史回复或业务
    事实下发给子 Agent。Supervisor 负责把指代消解进 assignment；这里只保留
    跨领域安全复用的用户偏好。实时业务事实必须由领域工具重新读取。
    """

    context = normalize_conversation_context(value)
    return ConversationContext(
        memories=[
            memory
            for memory in context.memories
            if memory.memory_type == MemoryType.PREFERENCE
        ]
    ).model_dump(mode="json")


def has_conversation_context(value: Any) -> bool:
    """判断规范化上下文是否实际包含摘要、消息或记忆。"""

    context = normalize_conversation_context(value)
    return bool(context.summary or context.recent_messages or context.memories)
