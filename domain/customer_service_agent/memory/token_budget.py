from __future__ import annotations

import math
import re
from typing import Sequence

from domain.customer_service_agent.memory.models import (
    ConversationMemoryMessage,
    MemoryPacket,
    MemoryReference,
)


class ConservativeTokenEstimator:
    """Provider-neutral fallback that deliberately favors over-counting."""

    _segments = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9]+|[^\s]")

    def count(self, text: str) -> int:
        total = 0
        for segment in self._segments.findall(text or ""):
            if len(segment) == 1 and "\u3400" <= segment <= "\u9fff":
                total += 1
            elif segment.isascii() and segment.isalnum():
                total += max(1, math.ceil(len(segment) / 3))
            else:
                total += 1
        return total

    def count_messages(self, messages: Sequence[ConversationMemoryMessage]) -> int:
        return sum(self.count(message.role) + self.count(message.content) + 2 for message in messages)

    def count_references(self, references: Sequence[MemoryReference]) -> int:
        return sum(self.count(item.memory_type.value) + self.count(item.content) + 2 for item in references)


class MemoryTokenBudgetAllocator:
    def __init__(self, estimator: ConservativeTokenEstimator | None = None) -> None:
        self.estimator = estimator or ConservativeTokenEstimator()

    def fit_text(self, text: str, budget: int) -> str:
        text = (text or "").strip()
        if not text or budget <= 0:
            return ""
        if self.estimator.count(text) <= budget:
            return text
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if self.estimator.count(text[:middle]) <= budget:
                low = middle
            else:
                high = middle - 1
        return text[:low].rstrip()

    def fit_messages(
        self,
        messages: Sequence[ConversationMemoryMessage],
        budget: int,
    ) -> list[ConversationMemoryMessage]:
        selected: list[ConversationMemoryMessage] = []
        remaining = max(0, budget)
        for message in reversed(messages):
            overhead = self.estimator.count(message.role) + 2
            content_budget = remaining - overhead
            if content_budget <= 0:
                continue
            content = self.fit_text(message.content, content_budget)
            if not content:
                continue
            fitted = message.model_copy(update={"content": content})
            cost = self.estimator.count_messages([fitted])
            if cost <= remaining:
                selected.insert(0, fitted)
                remaining -= cost
        return selected

    def fit_references(
        self,
        references: Sequence[MemoryReference],
        budget: int,
    ) -> list[MemoryReference]:
        selected: list[MemoryReference] = []
        remaining = max(0, budget)
        for reference in references:
            overhead = self.estimator.count(reference.memory_type.value) + 2
            content_budget = remaining - overhead
            if content_budget <= 0:
                break
            content = self.fit_text(reference.content, content_budget)
            if not content:
                continue
            fitted = reference.model_copy(update={"content": content})
            cost = self.estimator.count_references([fitted])
            if cost <= remaining:
                selected.append(fitted)
                remaining -= cost
        return selected

    def allocate(
        self,
        *,
        summary: str,
        recent_messages: Sequence[ConversationMemoryMessage],
        episodes: Sequence[MemoryReference],
        semantic_memories: Sequence[MemoryReference],
        recent_budget: int,
        summary_budget: int,
        episode_budget: int,
        semantic_budget: int,
        max_tokens: int,
    ) -> MemoryPacket:
        fitted_summary = self.fit_text(summary, min(summary_budget, max_tokens))
        fitted_recent = self.fit_messages(recent_messages, min(recent_budget, max_tokens))
        fitted_episodes = self.fit_references(episodes, min(episode_budget, max_tokens))
        fitted_semantic = self.fit_references(semantic_memories, min(semantic_budget, max_tokens))

        token_count = (
            self.estimator.count(fitted_summary)
            + self.estimator.count_messages(fitted_recent)
            + self.estimator.count_references(fitted_episodes)
            + self.estimator.count_references(fitted_semantic)
        )
        if token_count > max_tokens:
            overflow = token_count - max_tokens
            semantic_limit = max(0, self.estimator.count_references(fitted_semantic) - overflow)
            fitted_semantic = self.fit_references(fitted_semantic, semantic_limit)
            token_count = (
                self.estimator.count(fitted_summary)
                + self.estimator.count_messages(fitted_recent)
                + self.estimator.count_references(fitted_episodes)
                + self.estimator.count_references(fitted_semantic)
            )
        return MemoryPacket(
            session_summary=fitted_summary,
            recent_messages=fitted_recent,
            episodes=fitted_episodes,
            semantic_memories=fitted_semantic,
            token_count=token_count,
            max_tokens=max_tokens,
        )
