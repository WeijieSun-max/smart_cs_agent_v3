from __future__ import annotations

from typing import Any, Mapping, Sequence

from pydantic import ValidationError

from domain.customer_service_agent.memory.models import (
    ConversationContext,
    ConversationContextMemory,
    ConversationMemoryMessage,
    MemoryPacket,
)


def build_conversation_context(packet: MemoryPacket) -> ConversationContext:
    """Project a validated memory packet into the smaller LLM-facing contract."""

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
    return ConversationContext(recent_messages=list(messages))


def normalize_conversation_context(value: Any) -> ConversationContext:
    """Fail closed when graph state contains malformed or legacy context data."""

    if isinstance(value, ConversationContext):
        return value
    if not isinstance(value, Mapping):
        return ConversationContext()
    try:
        return ConversationContext.model_validate(value)
    except (TypeError, ValueError, ValidationError):
        return ConversationContext()


def conversation_context_payload(value: Any) -> dict[str, Any]:
    return normalize_conversation_context(value).model_dump(mode="json")


def has_conversation_context(value: Any) -> bool:
    context = normalize_conversation_context(value)
    return bool(context.summary or context.recent_messages or context.memories)
