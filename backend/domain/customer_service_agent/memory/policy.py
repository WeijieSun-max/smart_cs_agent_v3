"""决定候选记忆能否持久化的确定性安全与保留策略。"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta

from domain.customer_service_agent.memory.models import MemoryCandidate, MemoryType
from domain.customer_service_agent.policy.pii import detect_pii


@dataclass(frozen=True)
class MemoryPolicyConfig:
    """各记忆类型的最低置信度和默认保留天数。"""

    episode_min_confidence: float = 0.70
    preference_min_confidence: float = 0.85
    fact_min_confidence: float = 0.85
    task_min_confidence: float = 0.85
    episode_ttl_days: int = 180
    preference_ttl_days: int = 365
    fact_ttl_days: int = 180
    task_ttl_days: int = 90


@dataclass(frozen=True)
class MemoryPolicyDecision:
    """候选规范化、拒绝原因和过期时间的完整策略结果。"""

    accepted: bool
    memory_key: str
    reasons: tuple[str, ...]
    expires_at: datetime | None


_SENSITIVE_CREDENTIAL_PATTERN = re.compile(
    r"密码|口令|验证码|verification[ _-]?code|password|pin\b|cvv|安全码",
    re.IGNORECASE,
)
_INSTRUCTION_PATTERN = re.compile(
    r"忽略(?:之前|以上|所有).*指令|系统提示词|调用.*工具|转账工具|ignore previous|system prompt|call .*tool",
    re.IGNORECASE,
)
_DYNAMIC_KEY_PREFIXES = (
    "order.status",
    "order.logistics",
    "account.balance",
    "account.status",
    "ticket.status",
    "risk.level",
)


def normalize_memory_key(value: str) -> str:
    """把自由文本键规范化为稳定、可比较且有长度上限的索引键。"""

    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    normalized = re.sub(r"[\s:/\\]+", ".", normalized)
    normalized = re.sub(r"[^a-z0-9._\-㐀-鿿]", "", normalized)
    normalized = re.sub(r"\.{2,}", ".", normalized).strip(".")
    return normalized[:255]


class MemoryPolicy:
    """过滤 PII、凭证、提示注入、动态业务状态和低置信度候选。

    动态余额、订单状态等事实不得进入长期记忆，因为它们会过时且可能影响
    授权或写操作；这些数据必须每轮从权威业务工具重新读取。
    """

    def __init__(self, config: MemoryPolicyConfig) -> None:
        self.config = config

    def evaluate(self, candidate: MemoryCandidate, now: datetime) -> MemoryPolicyDecision:
        """一次性运行全部规则并返回所有拒绝原因，便于审计和指标统计。"""

        memory_key = normalize_memory_key(candidate.memory_key)
        reasons: list[str] = []
        serialized = json.dumps(candidate.structured_data, ensure_ascii=False, sort_keys=True)
        combined = f"{candidate.content}\n{serialized}"
        if not memory_key:
            reasons.append("invalid_memory_key")
        if detect_pii(combined):
            reasons.append("pii")
        if _SENSITIVE_CREDENTIAL_PATTERN.search(combined) or _SENSITIVE_CREDENTIAL_PATTERN.search(memory_key):
            reasons.append("sensitive_credential")
        if _INSTRUCTION_PATTERN.search(candidate.content):
            reasons.append("instruction_like_content")
        if candidate.memory_type == MemoryType.FACT and memory_key.startswith(_DYNAMIC_KEY_PREFIXES):
            reasons.append("dynamic_business_state")
        if candidate.confidence < self._threshold(candidate.memory_type):
            reasons.append("below_confidence_threshold")
        return MemoryPolicyDecision(
            accepted=not reasons,
            memory_key=memory_key,
            reasons=tuple(reasons),
            expires_at=now + timedelta(days=self._ttl_days(candidate.memory_type)),
        )

    def _threshold(self, memory_type: MemoryType) -> float:
        """返回该语义类型的最低接受置信度。"""

        return {
            MemoryType.EPISODE: self.config.episode_min_confidence,
            MemoryType.PREFERENCE: self.config.preference_min_confidence,
            MemoryType.FACT: self.config.fact_min_confidence,
            MemoryType.TASK: self.config.task_min_confidence,
        }[memory_type]

    def _ttl_days(self, memory_type: MemoryType) -> int:
        """返回该语义类型的默认保留期。"""

        return {
            MemoryType.EPISODE: self.config.episode_ttl_days,
            MemoryType.PREFERENCE: self.config.preference_ttl_days,
            MemoryType.FACT: self.config.fact_ttl_days,
            MemoryType.TASK: self.config.task_ttl_days,
        }[memory_type]
