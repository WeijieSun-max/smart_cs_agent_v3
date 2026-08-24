from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from domain.customer_service_agent.memory.models import (
    ConversationMemoryMessage,
    MemoryReference,
    MemoryType,
)
from domain.customer_service_agent.memory.token_budget import (
    ConservativeTokenEstimator,
    MemoryTokenBudgetAllocator,
)


def test_layer_budgets_and_total_budget_are_never_exceeded() -> None:
    estimator = ConservativeTokenEstimator()
    allocator = MemoryTokenBudgetAllocator(estimator)
    now = datetime.now(timezone.utc)
    messages = [ConversationMemoryMessage(role="user", content="中" * 900)]
    references = [
        MemoryReference(
            memory_id=uuid4(),
            memory_type=MemoryType.TASK,
            content="待办" * 500,
            confidence=0.95,
            score=0.95,
            updated_at=now,
        )
    ]

    packet = allocator.allocate(
        summary="摘要" * 500,
        recent_messages=messages,
        episodes=references,
        semantic_memories=references,
        recent_budget=700,
        summary_budget=350,
        episode_budget=350,
        semantic_budget=400,
        max_tokens=1800,
    )

    assert estimator.count(packet.session_summary) <= 350
    assert estimator.count_messages(packet.recent_messages) <= 700
    assert estimator.count_references(packet.episodes) <= 350
    assert estimator.count_references(packet.semantic_memories) <= 400
    assert packet.token_count <= 1800


def test_conservative_fallback_estimator_handles_chinese_and_ascii() -> None:
    estimator = ConservativeTokenEstimator()

    assert estimator.count("客户希望查询 order ABC-123") >= 8
    assert estimator.count("中" * 100) == 100
    assert estimator.count("") == 0


def test_high_priority_references_survive_budget_trimming() -> None:
    estimator = ConservativeTokenEstimator()
    allocator = MemoryTokenBudgetAllocator(estimator)
    now = datetime.now(timezone.utc)
    low = MemoryReference(
        memory_id=uuid4(),
        memory_type=MemoryType.FACT,
        content="普通事实" * 40,
        confidence=0.86,
        score=0.4,
        updated_at=now,
    )
    task = MemoryReference(
        memory_id=uuid4(),
        memory_type=MemoryType.TASK,
        content="跟进退款",
        confidence=0.99,
        score=1.0,
        updated_at=now,
    )

    selected = allocator.fit_references([task, low], 12)

    assert [item.memory_id for item in selected] == [task.memory_id]
