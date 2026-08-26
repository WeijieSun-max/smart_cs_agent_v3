from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from domain.customer_service_agent.orchestration.models import QueryUnderstandingResult, RouteDecision
from domain.customer_service_agent.orchestration.router import WRITE_CAPABILITIES, route_request
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.llm.llm_service import invoke_llm
from pkg.llm import parse_json_object
from pkg.log.logger import get_logger
from pkg.telemetry import record_json_parse

logger = get_logger()

_CAPABILITIES = frozenset({
    "plan_recommendation",
    "telecom_troubleshooting",
    "plan_change",
    "data_refuel",
    "roaming",
    "usage",
    "current_plan",
    "cancel_order",
    "update_order_address",
    "update_order_payment",
    "update_order_items",
    "request_return",
    "request_exchange",
    "price_adjustment_refund",
    "default_address",
    "order_query",
    "retail_policy",
    "product_query",
    "fallback",
})
_DOMAIN_VALUES = frozenset({"telecom", "retail", "fallback"})
_REFERENCE_PATTERN = re.compile(
    r"这个|那个|它|刚才|之前|上次|其中|第[一二三四五六七八九十\d]+个|昨天|前天|今天|上周|上个月|最近\d*天|买的",
    re.I,
)
_EXPLICIT_ID_PATTERN = re.compile(
    r"\b(line_id|plan_id|order_id|order_item_id|variant_id|address_id|payment_method_id)\s*[:=：]\s*([A-Za-z0-9_-]{1,64})",
    re.I,
)

UNDERSTANDING_SYSTEM_PROMPT = """你是电信与零售客服的查询理解器。结合当前问题和只读参考上下文，生成独立问题、业务域、能力、实体和歧义信息。
只允许能力：plan_recommendation、telecom_troubleshooting、plan_change、data_refuel、roaming、usage、current_plan、cancel_order、update_order_address、update_order_payment、update_order_items、request_return、request_exchange、price_adjustment_refund、default_address、order_query、retail_policy、product_query、fallback。
实体只提取当前用户明确给出的资源ID，以及 product_query、ordinal、status、amount、amount_gb、quantity、refund_method、reason。不得从参考上下文发明ID，不得把历史中的确认词当成当前确认。
只返回JSON：{"standalone_query":"...","domains":["retail"],"capabilities":["request_return"],"entities":{"product_query":"手机"},"ambiguity":true,"missing_fields":["order_id"],"requires_planning":false,"confidence":0.9}。
"""


class _UnderstandingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    standalone_query: str = Field(min_length=1, max_length=4000)
    domains: list[Literal["telecom", "retail", "fallback"]] = Field(min_length=1, max_length=3)
    capabilities: list[str] = Field(min_length=1, max_length=10)
    entities: dict[str, str] = Field(default_factory=dict)
    ambiguity: bool = False
    missing_fields: list[str] = Field(default_factory=list, max_length=20)
    requires_planning: bool = False
    confidence: float = Field(ge=0, le=1)


def query_understanding_node(state: ChatState) -> dict:
    result = understand_query(state)
    return {
        "normalized_query": result.standalone_query,
        "query_understanding": result.model_dump(mode="json"),
        "current_agent": "query_understanding",
        "node_logs": [f"查询理解完成：{result.source}"],
    }


def understand_query(state: ChatState) -> QueryUnderstandingResult:
    query = (state.get("raw_query") or "").strip()
    deterministic = route_request(query)
    if not _needs_llm(query, deterministic):
        return _from_route(query, deterministic, state.get("current_time") or "")
    try:
        response = invoke_llm(
            [
                SystemMessage(content=UNDERSTANDING_SYSTEM_PROMPT),
                HumanMessage(content=(
                    f"当前日期：{state.get('current_time') or ''}\n"
                    f"参考上下文：\n{(state.get('context_text') or '')[:6000]}\n\n"
                    f"当前用户问题：\n{query}"
                )),
            ],
            run_name="intent.understand",
            prompt_version="v1",
        )
        parsed = parse_json_object(str(response.content))
        decision = _UnderstandingDecision.model_validate(parsed)
        result = _validated_result(query, decision, state.get("current_time") or "")
        record_json_parse("intent.understand", True)
        return result
    except (ValidationError, TypeError, ValueError):
        record_json_parse("intent.understand", False)
    except Exception as exc:
        logger.warning("Query understanding degraded error_type={}", type(exc).__name__)
    fallback = _from_route(query, deterministic, state.get("current_time") or "")
    return fallback.model_copy(update={"source": "fallback"})


def route_from_understanding(state: ChatState) -> RouteDecision:
    data = state.get("query_understanding")
    if not data:
        return route_request((state.get("normalized_query") or state.get("raw_query") or "").strip())
    result = QueryUnderstandingResult.model_validate(data)
    capabilities = result.capabilities or ("fallback",)
    domains = result.domains or ("fallback",)
    return RouteDecision(
        domains=domains,
        capabilities=capabilities,
        confidence=result.confidence,
        composite=len(domains) > 1 or len(capabilities) > 1,
        risk_level="medium" if any(item in WRITE_CAPABILITIES for item in capabilities) else "low",
    )


def _needs_llm(query: str, decision: RouteDecision) -> bool:
    if re.fullmatch(r"(确认|确认执行|同意|是的|yes|confirm|取消|拒绝|不同意|不要|no|cancel)[。！!\s]*", query, re.I):
        return False
    return decision.confidence < 0.9 or decision.composite or bool(_REFERENCE_PATTERN.search(query))


def _from_route(query: str, route: RouteDecision, current_time: str) -> QueryUnderstandingResult:
    return QueryUnderstandingResult(
        standalone_query=query,
        domains=route.domains,
        capabilities=route.capabilities,
        entities=_explicit_entities(query),
        temporal_range=_resolve_temporal_range(query, current_time),
        ambiguity=False,
        requires_planning=_requires_planning(query, route.capabilities),
        confidence=route.confidence,
        source="deterministic",
    )


def _validated_result(query: str, decision: _UnderstandingDecision, current_time: str) -> QueryUnderstandingResult:
    capabilities = tuple(dict.fromkeys(item for item in decision.capabilities if item in _CAPABILITIES))
    domains = tuple(dict.fromkeys(item for item in decision.domains if item in _DOMAIN_VALUES))
    if not capabilities or not domains:
        raise ValueError("query understanding returned no supported route")
    return QueryUnderstandingResult(
        standalone_query=decision.standalone_query,
        domains=domains,
        capabilities=capabilities,
        entities=_sanitize_entities(query, decision.entities),
        temporal_range=_resolve_temporal_range(f"{query}\n{decision.standalone_query}", current_time),
        ambiguity=decision.ambiguity,
        missing_fields=tuple(dict.fromkeys(decision.missing_fields)),
        requires_planning=decision.requires_planning,
        confidence=decision.confidence,
        source="llm",
    )


def _explicit_entities(query: str) -> dict[str, str]:
    return {match.group(1).lower(): match.group(2) for match in _EXPLICIT_ID_PATTERN.finditer(query)}


def _sanitize_entities(query: str, entities: dict[str, str]) -> dict[str, str]:
    explicit = _explicit_entities(query)
    result = dict(explicit)
    for key in ("product_query", "ordinal", "status", "amount", "amount_gb", "quantity", "refund_method", "reason"):
        value = entities.get(key)
        if isinstance(value, str) and 0 < len(value) <= 255:
            result[key] = value
    return result


def _requires_planning(query: str, capabilities: tuple[str, ...]) -> bool:
    if not set(capabilities).intersection({"current_plan", "usage", "product_query"}):
        return False
    return bool(re.search(r"比较|综合|结合|分析|如果|并且|同时|为什么|怎么选", query))


def _resolve_temporal_range(query: str, current_time: str) -> dict[str, str] | None:
    try:
        today = date.fromisoformat(current_time) if current_time else datetime.now().date()
    except ValueError:
        today = datetime.now().date()
    start: date | None = None
    end: date | None = None
    if "前天" in query:
        start = today - timedelta(days=2)
        end = start + timedelta(days=1)
    elif "昨天" in query:
        start = today - timedelta(days=1)
        end = today
    elif "今天" in query:
        start = today
        end = today + timedelta(days=1)
    elif "本月" in query or "这个月" in query or "这月" in query:
        start = today.replace(day=1)
        end = (start + timedelta(days=32)).replace(day=1)
    elif match := re.search(r"最近\s*(\d{1,3})\s*天", query):
        days = min(365, max(1, int(match.group(1))))
        start = today - timedelta(days=days)
        end = today + timedelta(days=1)
    elif "上个月" in query:
        first_this_month = today.replace(day=1)
        end = first_this_month
        start = (first_this_month - timedelta(days=1)).replace(day=1)
    if start is None or end is None:
        return None
    return {"start": start.isoformat(), "end": end.isoformat()}
