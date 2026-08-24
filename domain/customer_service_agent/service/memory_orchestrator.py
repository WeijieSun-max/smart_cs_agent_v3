from __future__ import annotations

import re
from time import perf_counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Callable, Sequence

from domain.customer_service_agent.memory.models import (
    ConversationMemoryMessage,
    MemoryItem,
    MemoryPacket,
    MemoryReference,
    MemoryStatus,
    MemoryType,
    SessionSummary,
)
from domain.customer_service_agent.memory.token_budget import MemoryTokenBudgetAllocator
from pkg.log.logger import get_logger
from pkg.telemetry import memory_metrics


logger = get_logger()
_TYPE_PRIORITY = {
    MemoryType.TASK: 1.0,
    MemoryType.PREFERENCE: 0.9,
    MemoryType.FACT: 0.8,
    MemoryType.EPISODE: 0.7,
}


class MemoryOrchestrator:
    """Build a user-isolated, MySQL-validated and budgeted memory packet."""

    def __init__(
        self,
        *,
        repository,
        vector_index,
        embedder,
        short_term_memory,
        settings,
        allocator: MemoryTokenBudgetAllocator | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.vector_index = vector_index
        self.embedder = embedder
        self.short_term_memory = short_term_memory
        self.settings = settings
        self.allocator = allocator or MemoryTokenBudgetAllocator()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._diagnostics = {"candidates": 0, "hydrated": 0, "filtered": 0, "fallback": False}

    def build_packet(
        self,
        user_id: str,
        session_id: str,
        query: str,
        current_turn_id: str,
    ) -> MemoryPacket:
        started = perf_counter()
        self._diagnostics = {"candidates": 0, "hydrated": 0, "filtered": 0, "fallback": False}
        now = self.clock()
        summary = self._read_summary(user_id, session_id)
        recent = self._read_recent(user_id, session_id, current_turn_id)
        ranked = self._recall(user_id, query, now)
        episodes = [reference for item, reference in ranked if item.memory_type == MemoryType.EPISODE]
        semantic = [reference for item, reference in ranked if item.memory_type != MemoryType.EPISODE]
        packet = self.allocator.allocate(
            summary=summary,
            recent_messages=recent,
            episodes=episodes,
            semantic_memories=semantic,
            recent_budget=self.settings.memory_recent_messages_tokens,
            summary_budget=self.settings.memory_session_summary_tokens,
            episode_budget=self.settings.memory_episode_tokens,
            semantic_budget=self.settings.memory_semantic_tokens,
            max_tokens=self.settings.memory_context_max_tokens,
        )
        memory_metrics.record_packet(
            candidates=int(self._diagnostics["candidates"]),
            hydrated=int(self._diagnostics["hydrated"]),
            filtered=int(self._diagnostics["filtered"]),
            injected=(
                int(bool(packet.session_summary))
                + len(packet.recent_messages)
                + len(packet.episodes)
                + len(packet.semantic_memories)
            ),
            tokens=packet.token_count,
            latency_ms=(perf_counter() - started) * 1000,
            fallback=bool(self._diagnostics["fallback"]),
        )
        return packet

    def _read_summary(self, user_id: str, session_id: str) -> str:
        get_cached = getattr(self.short_term_memory, "get_session_summary", None)
        if callable(get_cached):
            try:
                cached = get_cached(session_id)
                if cached:
                    parsed = SessionSummary.model_validate(cached)
                    if parsed.user_id == user_id and parsed.session_id == session_id:
                        return parsed.summary_text
            except Exception as exc:
                logger.warning("Memory summary cache read degraded error_type={}", type(exc).__name__)
        try:
            summary = self.repository.get_latest_summary(user_id, session_id)
        except Exception as exc:
            logger.warning("Memory summary read degraded error_type={}", type(exc).__name__)
            return ""
        if summary is None:
            return ""
        cache_summary = getattr(self.short_term_memory, "cache_session_summary", None)
        if callable(cache_summary):
            try:
                cache_summary(session_id, summary.model_dump(mode="json"))
            except Exception as exc:
                logger.warning("Memory summary cache write degraded error_type={}", type(exc).__name__)
        return summary.summary_text

    def _read_recent(
        self,
        user_id: str,
        session_id: str,
        current_turn_id: str,
    ) -> list[ConversationMemoryMessage]:
        try:
            history = self.short_term_memory.get_history(session_id)
        except Exception as exc:
            logger.warning("Recent memory read degraded error_type={}", type(exc).__name__)
            try:
                history = self.repository.get_messages_after(user_id, session_id, 0, limit=100)
            except Exception:
                return []
        result: list[ConversationMemoryMessage] = []
        for message in history:
            if message.get("turn_id") == current_turn_id:
                continue
            role = str(message.get("role") or "")
            content = str(message.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            timestamp = _parse_timestamp(message.get("timestamp"))
            result.append(ConversationMemoryMessage(role=role, content=content[:8000], timestamp=timestamp))
        return result

    def _recall(self, user_id: str, query: str, now: datetime) -> list[tuple[MemoryItem, MemoryReference]]:
        score_by_id: dict[str, float] = {}
        candidates: Sequence[MemoryItem]
        try:
            if not self.vector_index or not self.vector_index.available or self.embedder is None:
                raise RuntimeError("semantic index unavailable")
            query_vector = self.embedder.embed_query(query)
            hits = self.vector_index.search(
                user_id=user_id,
                query_vector=query_vector,
                memory_types=list(MemoryType),
                top_k=self.settings.memory_recall_top_k,
                now=now,
            )
            minimum_score = getattr(self.settings, "memory_recall_min_score", 0.35)
            hits = [hit for hit in hits if hit.score >= minimum_score]
            memory_ids = [str(hit.memory_id) for hit in hits]
            self._diagnostics["candidates"] = len(hits)
            score_by_id = {str(hit.memory_id): hit.score for hit in hits}
            candidates = self.repository.get_items_by_ids(user_id, memory_ids, now) if memory_ids else []
            self._diagnostics["hydrated"] = len(candidates)
        except Exception as exc:
            self._diagnostics["fallback"] = True
            logger.warning("Semantic memory recall degraded error_type={}", type(exc).__name__)
            try:
                candidates = self.repository.list_active_items(
                    user_id,
                    memory_types=list(MemoryType),
                    limit=self.settings.memory_fallback_items,
                    now=now,
                )
                minimum_score = getattr(self.settings, "memory_recall_min_score", 0.35)
                candidates = [
                    item
                    for item in candidates
                    if item.memory_type == MemoryType.TASK
                    or _lexical_score(query, item.content) >= minimum_score
                ]
                self._diagnostics["candidates"] = len(candidates)
                self._diagnostics["hydrated"] = len(candidates)
            except Exception as fallback_exc:
                logger.warning("MySQL memory fallback failed error_type={}", type(fallback_exc).__name__)
                return []

        validated = [item for item in candidates if self._valid_for_user(item, user_id, now)]
        self._diagnostics["filtered"] = max(
            0,
            int(self._diagnostics["candidates"]) - len(validated),
        )
        scored = [
            (item, self._score(item, score_by_id.get(str(item.memory_id), _lexical_score(query, item.content)), now))
            for item in validated
        ]
        scored.sort(key=lambda pair: (-pair[1], -pair[0].updated_at.timestamp(), str(pair[0].memory_id)))
        deduplicated = self._deduplicate(scored)
        return [
            (
                item,
                MemoryReference(
                    memory_id=item.memory_id,
                    memory_type=item.memory_type,
                    content=item.content,
                    confidence=item.confidence,
                    score=score,
                    updated_at=item.updated_at,
                ),
            )
            for item, score in deduplicated
        ]

    @staticmethod
    def _valid_for_user(item: MemoryItem, user_id: str, now: datetime) -> bool:
        return bool(
            item.user_id == user_id
            and item.status == MemoryStatus.ACTIVE
            and (item.expires_at is None or item.expires_at > now)
            and (item.valid_from is None or item.valid_from <= now)
            and (item.valid_until is None or item.valid_until > now)
        )

    @staticmethod
    def _score(item: MemoryItem, similarity: float, now: datetime) -> float:
        age_days = max(0.0, (now - item.updated_at).total_seconds() / 86400)
        recency = 1.0 / (1.0 + age_days / 30.0)
        score = (
            0.55 * max(0.0, min(1.0, similarity))
            + 0.15 * recency
            + 0.15 * item.confidence
            + 0.15 * _TYPE_PRIORITY[item.memory_type]
        )
        return max(0.0, min(1.0, score))

    @staticmethod
    def _deduplicate(scored: Sequence[tuple[MemoryItem, float]]) -> list[tuple[MemoryItem, float]]:
        selected: list[tuple[MemoryItem, float]] = []
        keys: set[tuple[MemoryType, str]] = set()
        normalized_contents: list[str] = []
        for item, score in scored:
            key = (item.memory_type, item.memory_key)
            normalized = _normalize_content(item.content)
            if key in keys:
                continue
            if any(SequenceMatcher(None, normalized, existing).ratio() >= 0.92 for existing in normalized_contents):
                continue
            keys.add(key)
            normalized_contents.append(normalized)
            selected.append((item, score))
        return selected


def _parse_timestamp(value) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _normalize_content(value: str) -> str:
    return re.sub(r"\W+", "", value, flags=re.UNICODE).lower()


def _lexical_score(query: str, content: str) -> float:
    query_terms = set(re.findall(r"[\w\u3400-\u9fff]", query.lower()))
    content_terms = set(re.findall(r"[\w\u3400-\u9fff]", content.lower()))
    if not query_terms:
        return 0.0
    return len(query_terms & content_terms) / len(query_terms)
