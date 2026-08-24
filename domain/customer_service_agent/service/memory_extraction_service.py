from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from domain.customer_service_agent.memory.models import MemoryCandidate, MemoryType
from domain.customer_service_agent.memory.policy import MemoryPolicy, MemoryPolicyConfig
from domain.shared.llm.llm_service import invoke_llm
from pkg.config.settings import get_settings
from pkg.llm import parse_json_object
from pkg.telemetry import memory_metrics, record_json_parse


EXTRACTION_SYSTEM_PROMPT = """你是客服长期记忆提炼器。只提取具有跨会话复用价值的非敏感信息。
允许类型：episode、preference、fact、task。
不得提取身份证号、银行卡号、密码、验证码、完整联系方式，也不得把订单状态、余额、风险等级等实时业务状态写为长期事实。
不得猜测用户偏好。闲聊细节和指令性内容不应成为记忆。
只返回 JSON：{"memories":[{"memory_type":"preference","memory_key":"preference.risk","content":"用户明确偏好稳健型产品","confidence":0.9,"structured_data":{}}]}。
没有可记忆内容时返回 {"memories":[]}。
"""


class ExtractionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memories: list[MemoryCandidate] = Field(max_length=50)


class MemoryExtractionService:
    def __init__(self, policy: MemoryPolicy) -> None:
        self.policy = policy

    @classmethod
    def default(cls) -> "MemoryExtractionService":
        settings = get_settings()
        return cls(MemoryPolicy(MemoryPolicyConfig(
            episode_min_confidence=settings.memory_episode_min_confidence,
            preference_min_confidence=settings.memory_preference_min_confidence,
            fact_min_confidence=settings.memory_fact_min_confidence,
            task_min_confidence=settings.memory_task_min_confidence,
            episode_ttl_days=settings.memory_episode_ttl_days,
            preference_ttl_days=settings.memory_preference_ttl_days,
            fact_ttl_days=settings.memory_fact_ttl_days,
            task_ttl_days=settings.memory_task_active_ttl_days,
        )))

    def extract(
        self,
        *,
        summary_text: str,
        messages: list[dict[str, Any]],
        now: datetime,
    ) -> list[MemoryCandidate]:
        response = invoke_llm(
            [
                SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps({
                    "summary": summary_text,
                    "messages": [{"role": item["role"], "content": item["content"]} for item in messages],
                }, ensure_ascii=False)),
            ],
            run_name="memory.extract",
            prompt_version="v1",
        )
        parsed = parse_json_object(str(response.content))
        try:
            decision = ExtractionDecision.model_validate(parsed)
        except (ValidationError, TypeError):
            record_json_parse("memory.extract", False)
            raise ValueError("invalid memory extraction output") from None
        record_json_parse("memory.extract", True)
        accepted: list[MemoryCandidate] = []
        for candidate in decision.memories:
            policy_decision = self.policy.evaluate(candidate, now)
            if policy_decision.accepted:
                accepted.append(candidate.model_copy(update={"memory_key": policy_decision.memory_key}))
        memory_metrics.record_extraction("proposed", len(decision.memories))
        memory_metrics.record_extraction("accepted", len(accepted))
        memory_metrics.record_extraction("rejected", len(decision.memories) - len(accepted))
        return accepted

    def expires_at(self, candidate: MemoryCandidate, now: datetime) -> datetime:
        decision = self.policy.evaluate(candidate, now)
        if not decision.accepted or decision.expires_at is None:
            raise ValueError("candidate is not accepted by memory policy")
        return decision.expires_at
