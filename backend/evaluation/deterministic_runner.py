from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from domain.action_governance import GovernedActionService, initialize_action_service
from domain.business.service import initialize_service as initialize_business_service
from domain.business.store import InMemoryBusinessStore
from domain.customer_service_agent.file_skills import initialize_catalog
from domain.customer_service_agent.retrieval.answer_cache import rag_answer_cache
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
from evaluation.schema import EvaluationTask
from evaluation.simulators import ScriptedUserSimulator
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

    def bind_tools(self, _tools: Any, **_kwargs: Any) -> "ScriptedChatModel":
        """Accept native contracts while replaying legacy migration fixtures."""

        return self

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
        try:
            result = await super().execute_read(tool_name, arguments, identity, skill=skill)
        except Exception as exc:
            self._trajectory.append({
                "phase": "tool_result",
                "tool_name": tool_name,
                "effect": "read",
                "arguments": deepcopy(arguments),
                "success": False,
                "error_type": type(exc).__name__,
            })
            raise
        self._trajectory.append({
            "phase": "tool_result",
            "tool_name": tool_name,
            "effect": "read",
            "arguments": deepcopy(arguments),
            "success": True,
            "result": deepcopy(result),
        })
        return result

    def propose_write(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        identity: RequestIdentityContext,
        *,
        impact_summary: str,
        skill=None,
    ) -> Any:
        envelope = super().propose_write(
            tool_name,
            arguments,
            identity,
            impact_summary=impact_summary,
            skill=skill,
        )
        self._trajectory.append({
            "phase": "proposal",
            "tool_name": tool_name,
            "effect": "write",
            "arguments": deepcopy(envelope.arguments),
            "action_id": envelope.action_id,
            "status": envelope.status,
            "actor": "agent",
        })
        return envelope

    async def confirm(self, identity: RequestIdentityContext) -> Any:
        active = self.get_active(identity)
        if active is not None:
            self._trajectory.append({
                "phase": "confirmation",
                "tool_name": active.tool_name,
                "effect": "write",
                "arguments": deepcopy(active.arguments),
                "action_id": active.action_id,
                "actor": "user",
                "turn_id": identity.turn_id,
            })
        result = await super().confirm(identity)
        self._trajectory.append({
            "phase": "execution",
            "tool_name": result.tool_name,
            "effect": "write",
            "arguments": deepcopy(result.arguments),
            "action_id": result.action_id,
            "status": result.status,
            "receipt": deepcopy(result.receipt),
        })
        return result

    def reject(self, identity: RequestIdentityContext) -> Any:
        active = self.get_active(identity)
        result = super().reject(identity)
        self._trajectory.append({
            "phase": "rejection",
            "tool_name": active.tool_name if active is not None else result.tool_name,
            "effect": "write",
            "arguments": deepcopy(result.arguments),
            "action_id": result.action_id,
            "status": result.status,
            "actor": "user",
            "turn_id": identity.turn_id,
        })
        return result


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
_store: InMemoryBusinessStore | None = None
_trajectory: list[dict[str, Any]] = []


def bootstrap(*, fixtures: dict[str, list[dict[str, Any]]] | None = None) -> None:
    """一次性把整套确定性平台装好：LLM double、内存 checkpoint、业务数据、
    治理服务、Skill 目录、知识库、LangGraph 工作流。"""
    global _booted, _llm, _graph
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

    _reset_case_environment(fixtures or default_fixtures())

    initialize_catalog(_BACKEND_ROOT / "skills", get_mcp_server())
    initialize_knowledge_service(_build_knowledge_store())
    customer_service_workflow.initialize_workflow()
    _graph = customer_service_workflow.get_workflow()
    _booted = True


def run_case(case: EvaluationCase) -> RunOutcome:
    bootstrap()
    assert _llm is not None and _graph is not None
    fixtures = default_fixtures()
    for table, rows in case.initial_state_data.items():
        fixtures[table] = deepcopy(rows)
    _reset_case_environment(fixtures)
    assert _actions is not None and _store is not None
    _llm.reset(case.llm_script)
    environment_before = _store.evaluation_snapshot()

    run_id = uuid4().hex
    session_id = f"eval-{case.case_id}-{run_id[:8]}"
    state = create_chat_state(case.user_id, session_id, case.message, turn_id=f"eval-{run_id[:12]}")
    config = {"configurable": {"thread_id": f"{case.user_id}:{session_id}"}}

    started = time.perf_counter()
    final = _graph.invoke(state, config=config)
    elapsed_ms = (time.perf_counter() - started) * 1000
    _trajectory.append({
        "phase": "assistant_message",
        "turn_index": 0,
        "content": str(final.get("final_response") or ""),
    })

    identity = RequestIdentityContext(
        user_id=case.user_id,
        session_id=session_id,
        turn_id=f"eval-{run_id[:12]}",
    )
    try:
        active = _actions.get_active(identity)
        pending_action = active.action_id if active is not None else None
    except Exception:
        pending_action = None

    final_state = {**final, "pending_action": pending_action}
    return RunOutcome(
        run_id=run_id,
        trajectory=list(_trajectory),
        final_state=final_state,
        environment_before=environment_before,
        environment_after=_store.evaluation_snapshot(),
        turns=[{
            "turn_index": 0,
            "user_message": case.message,
            "assistant_message": str(final.get("final_response") or ""),
            "intent": final.get("intent"),
        }],
        llm_calls=len(_llm.calls),
        llm_call_runs=list(_llm.calls),
        elapsed_ms=elapsed_ms,
    )


def run_task_conversation(task: EvaluationTask) -> RunOutcome:
    """Run a versioned task across scripted user turns in one isolated session."""

    bootstrap()
    assert _llm is not None and _graph is not None
    fixtures = default_fixtures()
    for table, rows in task.initial_state.data.items():
        fixtures[table] = deepcopy(rows)
    _reset_case_environment(fixtures)
    assert _actions is not None and _store is not None

    run_id = uuid4().hex
    session_id = f"eval-{task.id}-{run_id[:8]}"
    thread_id = f"{task.user_id}:{session_id}"
    environment_before = _store.evaluation_snapshot()
    simulator = ScriptedUserSimulator()
    simulator.reset(task, environment_before)
    message = task.ticket
    script = task.llm_script
    max_turns = task.evaluation_criteria.budgets.max_turns or (
        1 + len(task.user_scenario.scripted_turns)
    )
    all_llm_calls: list[str] = []
    turn_results: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
    started = time.perf_counter()

    last_turn_index = 0
    for turn_index in range(max_turns):
        last_turn_index = turn_index
        _llm.reset(script)
        turn_id = f"eval-{run_id[:10]}-{turn_index}"
        state = create_chat_state(
            task.user_id,
            session_id,
            message,
            turn_id=turn_id,
        )
        final = _graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
        all_llm_calls.extend(_llm.calls)
        assistant_message = str(final.get("final_response") or "")
        _trajectory.append({
            "phase": "assistant_message",
            "turn_index": turn_index,
            "content": assistant_message,
        })
        turn_results.append({
            "turn_index": turn_index,
            "turn_id": turn_id,
            "user_message": message,
            "assistant_message": assistant_message,
            "intent": final.get("intent"),
        })
        user_decision = simulator.next_turn(
            assistant_message,
            _store.evaluation_snapshot(),
        )
        turn_results[-1]["simulator_stop_reason"] = user_decision.reason
        if user_decision.stop:
            break
        message = user_decision.message or ""
        script = user_decision.llm_script

    elapsed_ms = (time.perf_counter() - started) * 1000
    identity = RequestIdentityContext(
        user_id=task.user_id,
        session_id=session_id,
        turn_id=f"eval-{run_id[:10]}-{last_turn_index}",
    )
    try:
        active = _actions.get_active(identity)
        pending_action = active.action_id if active is not None else None
    except Exception:
        pending_action = None

    return RunOutcome(
        run_id=run_id,
        trajectory=list(_trajectory),
        final_state={**final, "pending_action": pending_action},
        environment_before=environment_before,
        environment_after=_store.evaluation_snapshot(),
        turns=turn_results,
        llm_calls=len(all_llm_calls),
        llm_call_runs=all_llm_calls,
        elapsed_ms=elapsed_ms,
    )


def _reset_case_environment(fixtures: dict[str, list[dict[str, Any]]]) -> None:
    """Install fresh business and governed-action state for exactly one case."""

    global _actions, _store
    _trajectory.clear()
    rag_answer_cache.clear()
    _store = InMemoryBusinessStore(deepcopy(fixtures))
    business = initialize_business_service(_store)
    _actions = RecordingActionService(
        get_mcp_server(),
        business,
        trajectory=_trajectory,
        ttl_seconds=900,
        tool_timeout_seconds=5,
    )
    initialize_action_service(_actions)
