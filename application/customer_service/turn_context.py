from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from adapter.web.schemas.chat import ChatRequest, ChatStreamRequest
from application.customer_service.session_ownership import get_session_ownership
from domain.customer_service_agent.memory.conversation_context import (
    build_recent_conversation_context,
)
from domain.customer_service_agent.memory.models import ConversationMemoryMessage
from domain.customer_service_agent.memory.token_budget import MemoryTokenBudgetAllocator
from domain.customer_service_agent.service import memory_service, short_term_memory_service
from domain.customer_service_agent.service.memory_orchestrator import MemoryOrchestrator
from domain.customer_service_agent.workflow.entity.chat_state import create_chat_state
from domain.shared.checkpoint import checkpoint_saver_service
from pkg.config.settings import Settings, get_settings
from pkg.exceptions.exception import RequestConflictError
from pkg.log.logger import get_logger
from pkg.security import get_local_user_id
from pkg.telemetry import create_turn_trace

logger = get_logger()


@dataclass(frozen=True)
class TurnContextDependencies:
    settings: Callable[[], Settings] = get_settings
    local_user_id: Callable[[], str] = get_local_user_id
    short_term_memory: Callable[[], Any] = short_term_memory_service.get_service
    memory_service: Callable[[], Any] = memory_service.get_service_or_none
    session_ownership: Callable[[], Any] = get_session_ownership
    checkpoint_service: Callable[[], Any] = lambda: checkpoint_saver_service.instance


DEFAULT_DEPENDENCIES = TurnContextDependencies()


def validate_turn_request(request: ChatRequest | ChatStreamRequest) -> None:
    # Contact details and addresses are valid customer-service business input.
    # Authentication credentials are protected at observability boundaries;
    # rejecting all PII here would make governed address/profile writes unusable.
    del request


def prepare_turn(request: ChatRequest | ChatStreamRequest, dependencies: TurnContextDependencies = DEFAULT_DEPENDENCIES):
    validate_turn_request(request)
    user_id = request.user_id or dependencies.local_user_id()
    initial_state = create_chat_state(user_id, request.session_id, request.message)
    session_id = initial_state["session_id"]
    dependencies.session_ownership().bind(user_id, session_id)
    turn = create_turn_trace(user_id, session_id, request.request_id)
    memory = dependencies.short_term_memory()
    replay = (
        memory.get_message_by_turn(turn.turn_id, "assistant", user_id=user_id)
        if request.request_id
        else None
    )
    existing_user = (
        memory.get_message_by_turn(turn.turn_id, "user", user_id=user_id)
        if request.request_id
        else None
    )
    for persisted in (replay, existing_user):
        persisted_session_id = persisted.get("session_id") if persisted is not None else None
        if persisted_session_id is not None and persisted_session_id != session_id:
            raise RequestConflictError()
    if existing_user is not None and existing_user.get("content") != request.message:
        raise RequestConflictError()
    settings = dependencies.settings()
    memory_packet = None
    conversation_context = None
    if replay is None:
        if settings.memory_layered_enabled:
            memory_packet = build_memory_packet(
                memory,
                user_id,
                session_id,
                request.message,
                turn.turn_id,
                settings,
                dependencies=dependencies,
            )
        if memory_packet is None:
            conversation_context = conversation_context_for_turn(
                memory,
                session_id,
                turn.turn_id,
                user_id=user_id,
                max_tokens=settings.memory_recent_messages_tokens,
            )
    pending_task = memory.get_pending_task(session_id, user_id=user_id)
    state = create_chat_state(
        user_id,
        session_id,
        request.message,
        turn_id=turn.turn_id,
        memory_packet=memory_packet,
        conversation_context=conversation_context,
        pending_task=pending_task,
        user_message_persisted=existing_user is not None,
    )
    return user_id, state, turn, replay


def build_memory_packet(
    short_term_memory,
    user_id: str,
    session_id: str,
    query: str,
    turn_id: str,
    settings: Settings,
    *,
    dependencies: TurnContextDependencies = DEFAULT_DEPENDENCIES,
) -> dict[str, Any] | None:
    service = dependencies.memory_service()
    if service is None:
        logger.warning("Layered memory is enabled but memory service is unavailable")
        return None
    orchestrator = MemoryOrchestrator(
        repository=service.repository,
        vector_index=service.vector_index,
        embedder=service.embedder,
        short_term_memory=short_term_memory,
        settings=settings,
    )
    try:
        packet = orchestrator.build_packet(user_id, session_id, query, turn_id)
    except Exception as exc:
        logger.warning("Memory packet construction failed error_type={}", type(exc).__name__)
        return None
    return packet.model_dump(mode="json")


def conversation_context_for_turn(
    memory,
    session_id: str,
    current_turn_id: str,
    *,
    user_id: str,
    max_tokens: int = 700,
) -> dict[str, Any]:
    history = memory.get_history(session_id, user_id=user_id)
    messages: list[ConversationMemoryMessage] = []
    for message in history:
        if message.get("turn_id") == current_turn_id:
            continue
        role = str(message.get("role") or "")
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        values: dict[str, Any] = {"role": role, "content": content[:8000]}
        if message.get("timestamp"):
            values["timestamp"] = message["timestamp"]
        try:
            messages.append(ConversationMemoryMessage.model_validate(values))
        except ValueError:
            values.pop("timestamp", None)
            messages.append(ConversationMemoryMessage.model_validate(values))

    fitted = MemoryTokenBudgetAllocator().fit_messages(messages, max_tokens)
    return build_recent_conversation_context(fitted).model_dump(mode="json")


def persist_assistant_turn(
    memory,
    session_id: str,
    content: str,
    turn_id: str,
    enqueue_memory: bool,
    user_id: str,
    lease=None,
) -> None:
    complete_turn = getattr(memory, "complete_turn", None)
    if callable(complete_turn):
        kwargs = {
            "enqueue_memory": enqueue_memory,
            "user_id": user_id,
        }
        if lease is not None and lease.fencing_token > 0:
            kwargs.update(
                fencing_token=lease.fencing_token,
                lease_owner_id=lease.owner_id,
            )
        complete_turn(
            session_id,
            content,
            turn_id,
            **kwargs,
        )
        return
    memory.add_message(session_id, "assistant", content, turn_id, user_id=user_id)


def persist_pending_task(
    memory,
    session_id: str,
    pending_task: dict[str, Any] | None,
    *,
    user_id: str,
) -> None:
    """把任务续接状态与自然语言消息分开保存，避免长回复裁剪关键槽位。"""

    if pending_task is None:
        memory.delete_pending_task(session_id, user_id=user_id)
    else:
        memory.cache_pending_task(session_id, pending_task, user_id=user_id)


def clear_checkpoint(
    user_id: str | None,
    session_id: str,
    checkpoint_ns: str | None = None,
    dependencies: TurnContextDependencies = DEFAULT_DEPENDENCIES,
) -> None:
    service = dependencies.checkpoint_service()
    if service is not None:
        service.clear_thread(user_id, session_id, checkpoint_ns=checkpoint_ns)
