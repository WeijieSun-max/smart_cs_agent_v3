from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from domain.customer_service_agent.orchestration.models import AgentResult
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.llm.llm_service import invoke_llm
from pkg.llm import parse_json_object
from pkg.telemetry import record_json_parse

_RESPONSE_SYSTEM_PROMPT = """你是客服回复整合器。只能重组给定 AgentResult 中的事实和用户片段，不得增加新事实、金额、状态、操作结果或设备能力。
保留引用标记、失败说明、限制条件和不确定性。不得把建议写成已执行结果。只返回JSON：{"response":"..."}。
用户主动提供、或当前用户有权读取的姓名、手机号、邮箱和收货地址允许按原文完整输出。
"""


class _ResponseDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    response: str = Field(min_length=1, max_length=20_000)


def response_writer_node(state: ChatState) -> dict[str, Any]:
    fallback = _fallback_text(state)
    results = _agent_results(state)
    supervisor_response = (state.get("supervisor_response") or "").strip()
    if supervisor_response:
        draft = supervisor_response
        source = state.get("supervisor_response_source") or "llm"
    elif not _should_compose(results):
        draft = fallback
        source = "deterministic"
    else:
        draft, composed = _compose(results, fallback)
        source = "llm" if composed else "fallback"
    # draft_source 供合规节点做风险分级：LLM 生成的内容仍需 llm_check，确定性模板可只走规则层。
    draft_source = "llm" if (source == "llm" or _results_have_llm_content(results)) else "deterministic"
    return {
        "draft_response": draft,
        "draft_source": draft_source,
        "current_agent": "response_writer",
        "node_logs": [f"Response draft completed: {source}"],
    }


def _results_have_llm_content(results: list[AgentResult]) -> bool:
    """RAG/ReAct 等单任务结果也含 LLM 生成的用户文本，需识别以保留合规 LLM 审查。"""
    return any(("rag" in (result.facts or {})) or ("observations" in (result.facts or {})) for result in results)


def _agent_results(state: ChatState) -> list[AgentResult]:
    results = []
    for value in state.get("task_results", {}).values():
        try:
            results.append(AgentResult.model_validate(value))
        except (ValidationError, TypeError):
            continue
    return results


def _should_compose(results: list[AgentResult]) -> bool:
    return len(results) > 1 and not any(result.status == "needs_confirmation" for result in results)


def _compose(results: list[AgentResult], fallback: str) -> tuple[str, bool]:
    payload = json.dumps([
        {
            "task_id": result.task_id,
            "status": result.status,
            "facts": result.facts,
            "user_fragment": result.user_fragment,
            "error_code": result.error_code,
        }
        for result in results
    ], ensure_ascii=False, default=str)
    try:
        response = invoke_llm(
            [
                SystemMessage(content=_RESPONSE_SYSTEM_PROMPT),
                HumanMessage(content=payload[:16_000]),
            ],
            run_name="response.compose",
            prompt_version="v1",
        )
        decision = _ResponseDecision.model_validate(parse_json_object(str(response.content)))
    except (ValidationError, TypeError):
        record_json_parse("response.compose", False)
        return fallback, False
    except Exception:
        return fallback, False
    record_json_parse("response.compose", True)
    return decision.response, True


def _fallback_text(state: ChatState) -> str:
    parts = [value for value in state.get("sub_results", {}).values() if isinstance(value, str)]
    return "\n\n".join(parts) if parts else "抱歉，暂时无法处理您的请求，请稍后重试。"
