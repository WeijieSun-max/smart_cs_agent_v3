from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from domain.action_governance import GovernedActionService, initialize_action_service
from domain.business.service import initialize_service as initialize_business_service
from domain.business.store import InMemoryBusinessStore
from domain.customer_service_agent.file_skills import initialize_catalog
from domain.customer_service_agent.service.knowledge_service import (
    initialize_service as initialize_knowledge_service,
)
from domain.customer_service_agent.tools.tool_registry import get_mcp_server
from domain.customer_service_agent.workflow import customer_service_workflow
from domain.customer_service_agent.workflow.entity.chat_state import create_chat_state
from domain.shared.checkpoint.checkpoint_saver_service import (
    initialize_service as initialize_checkpoint,
)
from domain.shared.identity import RequestIdentityContext
from domain.shared.llm.llm_service import initialize_llm_client
from evaluation.harness import EvaluationCase, RunOutcome
from infra.knowledge.local_knowledge_store import LocalKnowledgeStore

_BACKEND_ROOT = Path(__file__).resolve().parents[1]

_COMPLIANCE_PASS = '{"passed": true, "risk_level": "low", "violations": [], "suggestions": []}'


class ScriptedChatModel:
    """Deterministic, in-memory LLM double.

    按 ``run_name`` 顺序返回 case 脚本，同时记录每次调用，用于有界管理者轨迹和
    LLM 调用预算断言。
    """

    def __init__(self, responses: dict[str, str], *, default: str = ""):
        self._responses = responses
        self._default = default
        self.calls: list[str] = []
        self._script: dict[str, list[str]] = {}

    def invoke(self, messages: Any, config: dict[str, Any] | None = None, **kwargs: Any) -> AIMessage:
        run_name = "unknown"
        if isinstance(config, dict):
            metadata = config.get("metadata")
            run_name = (
                config.get("run_name")
                or (metadata or {}).get("prompt_name")
                or "unknown"
            )
        self.calls.append(run_name)
        scripted = self._script.get(run_name) or []
        content = scripted.pop(0) if scripted else self._responses.get(run_name, self._default)
        return AIMessage(content=content)

    def reset(self, script: dict[str, tuple[dict[str, Any] | str, ...]] | None = None) -> None:
        self.calls.clear()
        self._script = {
            run_name: [
                value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                for value in values
            ]
            for run_name, values in (script or {}).items()
        }


class RecordingActionService(GovernedActionService):
    """GovernedActionService 的子类，额外记录每个业务工具调用，构成 trajectory。"""

    def __init__(self, server, business, *, trajectory: list[dict[str, Any]], **kwargs: Any):
        super().__init__(server, business, **kwargs)
        self._trajectory = trajectory

    async def execute_read(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        identity: RequestIdentityContext,
        *,
        skill=None,
    ) -> Any:
        self._trajectory.append({"tool_name": tool_name, "effect": "read"})
        return await super().execute_read(tool_name, arguments, identity, skill=skill)

    def propose_write(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        identity: RequestIdentityContext,
        *,
        impact_summary: str,
        skill=None,
    ) -> Any:
        self._trajectory.append({"tool_name": tool_name, "effect": "write"})
        return super().propose_write(
            tool_name,
            arguments,
            identity,
            impact_summary=impact_summary,
            skill=skill,
        )


def default_fixtures() -> dict[str, list[dict[str, Any]]]:
    """与演示数据对齐的确定性业务 fixture（InMemoryBusinessStore 使用）。"""
    return {
        "users": [
            {"user_id": "u-active", "status": "active"},
            {"user_id": "u-other", "status": "active"},
        ],
        "accounts": [
            {"account_id": "a-active", "user_id": "u-active", "status": "active"},
            {"account_id": "a-other", "user_id": "u-other", "status": "active"},
        ],
        "lines": [
            {
                "line_id": "L1",
                "account_id": "a-active",
                "status": "active",
                "current_plan_id": "P1",
                "version": 1,
                "roaming_enabled": False,
            },
            {
                "line_id": "L-OTHER",
                "account_id": "a-other",
                "status": "active",
                "current_plan_id": "P1",
                "version": 1,
                "roaming_enabled": False,
            },
        ],
        "plans": [
            {
                "plan_id": "P1",
                "name": "畅享20G-59元套餐",
                "status": "active",
                "monthly_price": "59",
                "currency": "CNY",
                "data_limit_mb": 20480,
                "included_voice_minutes": 100,
                "data_unlimited": False,
                "voice_unlimited": False,
                "voice_overage_price_per_minute": "0.15",
                "refuel_price_per_gb": "10",
            },
            {
                "plan_id": "P2",
                "name": "畅享50G-89元套餐",
                "status": "active",
                "monthly_price": "89",
                "currency": "CNY",
                "data_limit_mb": 51200,
                "included_voice_minutes": 500,
                "data_unlimited": False,
                "voice_unlimited": False,
                "voice_overage_price_per_minute": "0.15",
                "refuel_price_per_gb": "10",
            },
        ],
        "usage_cycles": [
            {"usage_id": "U1", "line_id": "L1", "cycle_start": "2026-06-01", "used_data_mb": 30000, "used_voice_minutes": 300, "current": False, "refuel_count": 0, "elapsed_days": 30, "cycle_days": 30},
            {"usage_id": "U2", "line_id": "L1", "cycle_start": "2026-07-01", "used_data_mb": 32000, "used_voice_minutes": 320, "current": False, "refuel_count": 0, "elapsed_days": 30, "cycle_days": 30},
            {"usage_id": "U3", "line_id": "L1", "cycle_start": "2026-08-01", "used_data_mb": 31000, "used_voice_minutes": 310, "current": False, "refuel_count": 0, "elapsed_days": 30, "cycle_days": 30},
        ],
        "orders": [
            {"order_id": "O1", "order_no": "DEMO202608000001", "user_id": "u-active", "status": "pending", "grand_total": "599", "currency": "CNY", "version": 1, "placed_at": "2026-08-15T10:00:00"},
            {"order_id": "OTHER", "order_no": "OTHER999", "user_id": "u-other", "status": "pending", "grand_total": "199", "currency": "CNY", "version": 1, "placed_at": "2026-08-10T10:00:00"},
        ],
        "order_items": [
            {"order_item_id": "OI1", "order_id": "O1", "variant_id": "V1", "name_snapshot": "5G随身WiFi Pro", "sku_snapshot": "WIFI-5G", "quantity": 1, "returned_qty": 0, "exchanged_qty": 0},
        ],
        "products": [
            {"product_id": "PR1", "name": "5G随身WiFi Pro", "description": "便携5G路由", "status": "active"},
            {"product_id": "PR2", "name": "千兆双频路由器", "description": "家用路由", "status": "active"},
        ],
        "variants": [
            {"variant_id": "V1", "product_id": "PR1", "status": "active"},
        ],
        "inventory": [
            {"variant_id": "V1", "available_qty": 100},
        ],
    }


def _build_knowledge_store() -> LocalKnowledgeStore:
    store = LocalKnowledgeStore(
        _BACKEND_ROOT / "data" / "evaluation_knowledge_index",
        embedding_provider=None,
        keyword_min_score=0.18,
        candidate_limit=8,
    )
    store.add_document(
        "手机没有信号排查：1. 确认未开启飞行模式；2. 重启手机；3. 检查SIM卡是否正确插入；"
        "4. 在信号弱区域尝试移动位置；5. 若仍无信号，请联系人工客服。",
        source="telecom-troubleshooting",
        metadata={"domain": "telecom", "document_type": "troubleshooting", "version": "1.0", "status": "active"},
    )
    return store


_booted = False
_llm: ScriptedChatModel | None = None
_graph = None
_actions: RecordingActionService | None = None
_trajectory: list[dict[str, Any]] = []


def bootstrap(*, fixtures: dict[str, list[dict[str, Any]]] | None = None) -> None:
    """一次性把整套确定性平台装好：LLM double、内存 checkpoint、业务数据、
    治理服务、Skill 目录、知识库、LangGraph 工作流。"""
    global _booted, _llm, _graph, _actions
    if _booted:
        return
    _llm = ScriptedChatModel(
        {
            "compliance.review": _COMPLIANCE_PASS,
            "knowledge.answer": "根据知识库，请先确认飞行模式已关闭并重启设备；若仍无信号请联系人工。[1]",
            "response.compose": '{"response":"已为您汇总查询结果。"}',
        },
        default="",
    )
    initialize_llm_client(_llm)  # type: ignore[arg-type]
    initialize_checkpoint(MemorySaver())

    business = initialize_business_service(InMemoryBusinessStore(fixtures or default_fixtures()))
    _actions = RecordingActionService(
        get_mcp_server(),
        business,
        trajectory=_trajectory,
        ttl_seconds=900,
        tool_timeout_seconds=5,
    )
    initialize_action_service(_actions)

    initialize_catalog(_BACKEND_ROOT / "skills", get_mcp_server())
    initialize_knowledge_service(_build_knowledge_store())
    customer_service_workflow.initialize_workflow()
    _graph = customer_service_workflow.get_workflow()
    _booted = True


def run_case(case: EvaluationCase) -> RunOutcome:
    bootstrap()
    assert _llm is not None and _actions is not None and _graph is not None
    _llm.reset(case.llm_script)
    _trajectory.clear()

    state = create_chat_state(case.user_id, case.case_id, case.message)
    config = {"configurable": {"thread_id": f"{case.user_id}:{case.case_id}"}}

    started = time.perf_counter()
    final = _graph.invoke(state, config=config)
    elapsed_ms = (time.perf_counter() - started) * 1000

    identity = RequestIdentityContext(
        user_id=case.user_id,
        session_id=case.case_id,
        turn_id="eval",
    )
    try:
        active = _actions.get_active(identity)
        pending_action = active.action_id if active is not None else None
    except Exception:
        pending_action = None

    final_state = {**final, "pending_action": pending_action}
    return RunOutcome(
        trajectory=list(_trajectory),
        final_state=final_state,
        llm_calls=len(_llm.calls),
        llm_call_runs=list(_llm.calls),
        elapsed_ms=elapsed_ms,
    )
