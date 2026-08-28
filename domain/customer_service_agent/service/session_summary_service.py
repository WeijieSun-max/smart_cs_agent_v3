"""增量生成无敏感信息的版本化会话摘要。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from domain.customer_service_agent.memory.models import SessionSummary
from domain.customer_service_agent.policy.pii import detect_pii
from domain.shared.llm.llm_service import invoke_llm
from pkg.llm import parse_json_object
from pkg.telemetry import record_json_parse


SUMMARY_SYSTEM_PROMPT = """你是客服会话摘要器。请基于旧摘要和新增消息生成可增量更新的结构化摘要。
只返回 JSON，字段必须为：summary_text、goals、entities、actions、results、missing_info、pending_tasks。
保留用户目标、已确认实体、已执行动作、处理结果、缺失信息和未完成事项。
不得猜测，不得加入身份证号、银行卡号、密码、验证码等敏感信息。
"""


class SummaryDecision(BaseModel):
    """模型摘要输出的严格文本与结构化字段。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    summary_text: str = Field(min_length=1, max_length=12_000)
    goals: list[str] = Field(default_factory=list, max_length=50)
    entities: dict[str, str] = Field(default_factory=dict)
    actions: list[str] = Field(default_factory=list, max_length=100)
    results: list[str] = Field(default_factory=list, max_length=100)
    missing_info: list[str] = Field(default_factory=list, max_length=100)
    pending_tasks: list[str] = Field(default_factory=list, max_length=100)


class SessionSummaryService:
    """把旧摘要和新增消息压缩为下一版本，并执行 PII 否决。"""

    def build(
        self,
        *,
        user_id: str,
        session_id: str,
        previous: SessionSummary | None,
        new_messages: list[dict[str, Any]],
    ) -> SessionSummary:
        """生成覆盖到最大 message_id 的新摘要版本。

        空增量、非法 JSON/Schema 或任何结构字段中的 PII 都会导致失败；旧摘要
        不会被原地修改，worker 可安全重试同一消息范围。
        """

        if not new_messages:
            raise ValueError("summary requires new messages")
        response = invoke_llm(
            [
                SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps({
                    "previous_summary": previous.summary_text if previous else "",
                    "previous_structured_data": previous.structured_data if previous else {},
                    "new_messages": [
                        {"message_id": item["message_id"], "role": item["role"], "content": item["content"]}
                        for item in new_messages
                    ],
                }, ensure_ascii=False)),
            ],
            run_name="memory.session_summary",
            prompt_version="v1",
        )
        parsed = parse_json_object(str(response.content))
        try:
            decision = SummaryDecision.model_validate(parsed)
        except (ValidationError, TypeError):
            record_json_parse("memory.session_summary", False)
            raise ValueError("invalid summary output") from None
        if detect_pii(decision.summary_text) or detect_pii(json.dumps(decision.model_dump(), ensure_ascii=False)):
            raise ValueError("invalid summary output: sensitive content")
        record_json_parse("memory.session_summary", True)
        now = datetime.now(timezone.utc)
        return SessionSummary(
            user_id=user_id,
            session_id=session_id,
            version=(previous.version + 1) if previous else 1,
            summary_text=decision.summary_text,
            structured_data=decision.model_dump(exclude={"summary_text"}),
            covers_until_message_id=max(int(item["message_id"]) for item in new_messages),
            created_at=now,
            updated_at=now,
        )
