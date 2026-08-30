"""把 Supervisor/领域 Agent 结果整理为待合规审查的单一草稿。"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from domain.customer_service_agent.orchestration.models import AgentResult
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.llm.llm_service import ainvoke_llm, invoke_llm
from pkg.llm import parse_json_object
from pkg.telemetry import record_json_parse

_RESPONSE_SYSTEM_PROMPT = """你是客服回复整合器。只能重组给定 AgentResult 中的事实和用户片段，不得增加新事实、金额、状态、操作结果或设备能力。
保留引用标记、失败说明、限制条件和不确定性。不得把建议写成已执行结果。只返回JSON：{"response":"..."}。
用户主动提供、或当前用户有权读取的姓名、手机号、邮箱和收货地址允许按原文完整输出。
"""

_COMPOSITE_RESPONSE_PROMPT = """
复合请求中，各 AgentResult 仅代表对应领域。若某段声称另一个领域无法处理，但另一个 succeeded 结果已经提供该领域的工具事实，应省略前者的越界免责声明，只汇总成功事实与真实失败。不得输出互相矛盾的能力说明。
"""


class _ResponseDecision(BaseModel):
    """多结果 LLM 合成器的最小输出契约。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    response: str = Field(min_length=1, max_length=20_000)


def response_writer_node(state: ChatState) -> dict[str, Any]:
    """选择 Supervisor 直答、确定性拼接或受约束的 LLM 多结果合成。

    节点不会产出 `final_response`；它同时标记草稿是否包含 LLM 内容，供下一
    个不可绕过合规节点决定审查强度。合成失败时保留已有用户片段。
    """

    fallback = _fallback_text(state)
    results = _agent_results(state)
    supervisor_response = (state.get("supervisor_response") or "").strip()
    # 多领域结果始终经过专用 writer 重新汇总，避免 Supervisor 的通用复核回复
    # 直接带出某个领域 Agent 对兄弟领域的越界评论。
    if _should_compose(results):
        draft, composed = _compose(results, fallback)
        source = "llm" if composed else "fallback"
    elif supervisor_response:
        draft = supervisor_response
        source = state.get("supervisor_response_source") or "llm"
    else:
        draft = fallback
        source = "deterministic"
    # draft_source 供合规节点做风险分级：LLM 生成的内容仍需 llm_check，确定性模板可只走规则层。
    draft_source = "llm" if (source == "llm" or _results_have_llm_content(results)) else "deterministic"
    return {
        "draft_response": draft,
        "draft_source": draft_source,
        "current_agent": "response_writer",
        "node_logs": [f"Response draft completed: {source}"],
    }


async def response_writer_node_async(state: ChatState) -> dict[str, Any]:
    """Async graph entrypoint; synchronous entrypoint remains for offline callers."""

    fallback = _fallback_text(state)
    results = _agent_results(state)
    supervisor_response = (state.get("supervisor_response") or "").strip()
    if _should_compose(results):
        draft, composed = await _compose_async(results, fallback)
        source = "llm" if composed else "fallback"
    elif supervisor_response:
        draft = supervisor_response
        source = state.get("supervisor_response_source") or "llm"
    else:
        draft = fallback
        source = "deterministic"
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
    """忽略畸形任务值，只返回通过领域协议校验的结果。"""

    results = []
    for value in state.get("task_results", {}).values():
        try:
            results.append(AgentResult.model_validate(value))
        except (ValidationError, TypeError):
            continue
    return results


def _should_compose(results: list[AgentResult]) -> bool:
    """仅在多个结果且没有待确认动作时使用 LLM 重组。"""

    return len(results) > 1 and not any(result.status == "needs_confirmation" for result in results)


def _compose(results: list[AgentResult], fallback: str) -> tuple[str, bool]:
    """让模型只重组给定事实；输出非法或调用失败时返回确定性兜底。"""

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
                SystemMessage(content=_RESPONSE_SYSTEM_PROMPT + _COMPOSITE_RESPONSE_PROMPT),
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


async def _compose_async(results: list[AgentResult], fallback: str) -> tuple[str, bool]:
    payload = _composition_payload(results)
    try:
        response = await ainvoke_llm(
            [
                SystemMessage(content=_RESPONSE_SYSTEM_PROMPT + _COMPOSITE_RESPONSE_PROMPT),
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


def _composition_payload(results: list[AgentResult]) -> str:
    return json.dumps([
        {
            "task_id": result.task_id,
            "status": result.status,
            "facts": result.facts,
            "user_fragment": result.user_fragment,
            "error_code": result.error_code,
        }
        for result in results
    ], ensure_ascii=False, default=str)


def _fallback_text(state: ChatState) -> str:
    """按现有子结果构建不会新增事实的兜底文本。"""

    parts = [value for value in state.get("sub_results", {}).values() if isinstance(value, str)]
    return "\n\n".join(parts) if parts else "抱歉，暂时无法处理您的请求，请稍后重试。"
