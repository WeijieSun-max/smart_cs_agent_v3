from __future__ import annotations

import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from domain.action_governance import GovernedActionService, initialize_action_service
from domain.business.service import initialize_service
from domain.business.store import InMemoryBusinessStore
from domain.customer_service_agent.agents import knowledge_agent, supervisor_agent, tool_agent
from domain.customer_service_agent.orchestration.models import (
    AgentAssignment,
    AgentResult,
    PendingWritePlan,
    SupervisorDecision,
)
from domain.customer_service_agent.orchestration.pending_action_resolver import _action_result_text
from domain.customer_service_agent.tools.tool_registry import get_mcp_server
from domain.customer_service_agent.workflow.entity.chat_state import create_chat_state
from domain.customer_service_agent.workflow.nodes import supervisor_graph_nodes
from domain.shared.identity import RequestIdentityContext


class Response:
    def __init__(self, content: str) -> None:
        self.content = content


def _platform(*, addresses=None):
    store = InMemoryBusinessStore({
        "users": [{"user_id": "u1", "status": "active"}],
        "accounts": [{"account_id": "a1", "user_id": "u1", "status": "active"}],
        "lines": [{
            "line_id": "L1",
            "account_id": "a1",
            "status": "active",
            "current_plan_id": "P1",
            "version": 1,
            "roaming_enabled": False,
        }],
        "plans": [
            {
                "plan_id": "P1",
                "name": "基础套餐",
                "status": "active",
                "monthly_price": "80",
                "currency": "CNY",
                "data_limit_mb": 10240,
                "included_voice_minutes": 100,
            },
            {
                "plan_id": "P2",
                "name": "畅享套餐",
                "status": "active",
                "monthly_price": "90",
                "currency": "CNY",
                "data_limit_mb": 30720,
                "included_voice_minutes": 500,
            },
        ],
        "addresses": list(addresses or []),
    })
    business = initialize_service(store)
    actions = GovernedActionService(get_mcp_server(), business)
    initialize_action_service(actions)
    return store, actions


def _identity(session_id: str = "s1") -> RequestIdentityContext:
    return RequestIdentityContext(user_id="u1", session_id=session_id, turn_id="t1")


def test_telecom_agent_generates_proposal_but_does_not_execute(monkeypatch) -> None:
    store, actions = _platform()
    responses = iter([
        Response(
            '{"action":"tool_call","tool_name":"telecom_quote_plan_change",'
            '"arguments":{"line_id":"L1","plan_id":"P2"}}'
        ),
        Response(
            '{"action":"propose_write","tool_name":"telecom_change_plan",'
            '"arguments":{"line_id":"L1","plan_id":"P2","expected_version":1},'
            '"impact_summary":"将线路 L1 从基础套餐变更为畅享套餐，月租由80变为90 CNY"}'
        ),
    ])
    monkeypatch.setattr(tool_agent, "invoke_llm", lambda *_args, **_kwargs: next(responses))
    assignment = AgentAssignment(
        task_id="T1",
        agent="telecom_agent",
        objective="把线路 L1 变更为套餐 P2",
        capability="plan_change",
        arguments={"line_id": "L1", "plan_id": "P2"},
    )

    result = asyncio.run(tool_agent.run_tool_agent(
        assignment,
        create_chat_state("u1", "s1", "把 L1 换成 P2"),
        _identity(),
    ))

    assert result.status == "needs_confirmation"
    assert result.pending_action_id
    assert store.get_owned("lines", "L1", "u1")["current_plan_id"] == "P1"
    completed = asyncio.run(actions.confirm(_identity()))
    assert completed.status == "succeeded"
    assert store.get_owned("lines", "L1", "u1")["current_plan_id"] == "P2"


def test_retail_agent_creates_full_address_and_sets_it_default_after_confirmation(monkeypatch) -> None:
    store, actions = _platform()
    responses = iter([
        Response(
            '{"action":"tool_call","tool_name":"retail_list_addresses",'
            '"arguments":{}}'
        ),
        Response(
            '{"action":"propose_write","tool_name":"retail_create_address",'
            '"arguments":{"recipient":"张伟","phone":"18060815554",'
            '"province":"江苏省","city":"南京市","district":"栖霞区",'
            '"detail":"文艺路9号南京邮电大学仙林校区东门","set_default":true},'
            '"impact_summary":"新增张伟的南京收货地址并设为默认地址，联系电话18060815554"}'
        ),
    ])
    monkeypatch.setattr(tool_agent, "invoke_llm", lambda *_args, **_kwargs: next(responses))
    assignment = AgentAssignment(
        task_id="R1",
        agent="retail_agent",
        objective="新增南京地址并设为默认地址",
        capability="default_address",
    )

    result = asyncio.run(tool_agent.run_tool_agent(
        assignment,
        create_chat_state(
            "u1",
            "s1",
            "张伟 18060815554 江苏省南京市栖霞区文艺路9号南京邮电大学仙林校区东门",
        ),
        _identity(),
    ))

    assert result.status == "needs_confirmation"
    assert store.list_owned("addresses", "u1", status="active") == []
    completed = asyncio.run(actions.confirm(_identity()))
    assert completed.status == "succeeded"
    addresses = actions.business.list_addresses("u1")
    assert len(addresses) == 1
    assert addresses[0]["recipient"] == "张伟"
    assert addresses[0]["phone"] == "18060815554"
    assert addresses[0]["detail"] == "文艺路9号南京邮电大学仙林校区东门"
    assert addresses[0]["is_default"] is True
    stored = store.list_owned("addresses", "u1", status="active")[0]
    assert b"18060815554" not in stored["phone_cipher"]


def test_retail_agent_reuses_current_default_contact_without_asking_again(monkeypatch) -> None:
    store, actions = _platform(addresses=[{
        "address_id": "A1",
        "user_id": "u1",
        "label": "原默认地址",
        "recipient": "苏军",
        "phone": "15588697856",
        "province": "江苏省",
        "city": "南京市",
        "district": "栖霞区",
        "detail": "原地址1号",
        "is_default": True,
        "status": "active",
        "version": 3,
    }])
    captured_messages = []
    captured_tools = []
    responses = iter([
        AIMessage(content="", tool_calls=[{
            "id": "call-create-reusing-contact",
            "name": "retail_create_address_reusing_default_contact",
            "args": {
                "province": "北京市",
                "city": "北京市",
                "district": "朝阳区",
                "detail": "应天路88号",
                "set_default": True,
                "impact_summary": "沿用当前默认联系人，创建北京地址并设为默认地址",
            },
        }]),
    ])

    def invoke(messages, **kwargs):
        captured_messages.append(messages)
        captured_tools.append(kwargs["tools"])
        return next(responses)

    monkeypatch.setattr(tool_agent, "invoke_llm", invoke)
    assignment = AgentAssignment(
        task_id="R-reuse",
        agent="retail_agent",
        objective="将默认地址改为北京市朝阳区应天路88号并沿用原联系人",
        capability="default_address",
        arguments={"contact_strategy": "reuse_current_default"},
    )

    result = asyncio.run(tool_agent.run_tool_agent(
        assignment,
        create_chat_state("u1", "s1", assignment.objective),
        _identity(),
    ))

    assert result.status == "needs_confirmation"
    assignment_payload = json.loads(captured_messages[0][1].content)
    assert assignment_payload["type"] == "agent_assignment"
    assert "observations" not in assignment_payload
    assert len(captured_messages[0]) == 2
    exposed_names = {
        item["function"]["name"]
        for item in captured_tools[0]
    }
    assert "retail_create_address_reusing_default_contact" in exposed_names
    assert "retail_create_address" not in exposed_names
    assert "retail_get_default_address" not in exposed_names
    proposal = result.facts["proposal"]
    assert proposal["tool_name"] == "retail_create_address_reusing_default_contact"
    assert proposal["tool_call_id"] == "call-create-reusing-contact"
    pending = actions.get_active(_identity())
    assert pending is not None
    assert "recipient" not in pending.arguments
    assert "phone" not in pending.arguments
    completed = asyncio.run(actions.confirm(_identity()))
    assert completed.status == "succeeded"
    assert completed.receipt["summary"]["full_address"] == "北京市朝阳区应天路88号"
    default_address = actions.business.get_default_address("u1")["address"]
    assert default_address["recipient"] == "苏军"
    assert default_address["phone"] == "15588697856"
    assert default_address["province"] == "北京市"
    assert default_address["district"] == "朝阳区"
    assert default_address["detail"] == "应天路88号"
    assert store.get_owned("addresses", "A1", "u1")["is_default"] is False
    user_text = _action_result_text(
        completed.status,
        completed.impact_summary,
        completed.receipt,
    )
    assert "北京市北京市" not in user_text
    assert "执行回执已生成" in user_text
    assert "resource_id" not in user_text


def test_default_address_recognizes_mysql_tinyint_flag() -> None:
    _store, actions = _platform(addresses=[{
        "address_id": "A-mysql-bool",
        "user_id": "u1",
        "label": "默认地址",
        "recipient": "苏军",
        "phone": "15588697856",
        "province": "江苏省",
        "city": "南京市",
        "district": "栖霞区",
        "detail": "原地址1号",
        "is_default": 1,
        "status": "active",
        "version": 3,
    }])

    result = actions.business.get_default_address("u1")

    assert result["status"] == "found"
    assert result["address"]["address_id"] == "A-mysql-bool"


def test_retail_agent_uses_high_level_tool_for_nested_default_address_change(monkeypatch) -> None:
    store, actions = _platform(addresses=[{
        "address_id": "A1",
        "user_id": "u1",
        "label": "原默认地址",
        "recipient": "苏军",
        "phone": "15588697856",
        "province": "江苏省",
        "city": "南京市",
        "district": "栖霞区",
        "detail": "原地址1号",
        "is_default": True,
        "status": "active",
        "version": 3,
    }])
    responses = iter([
        AIMessage(content="", tool_calls=[{
            "id": "call-nested-address",
            "name": "retail_create_address_reusing_default_contact",
            "args": {
                "province": "河南省",
                "city": "郑州市",
                "district": "二七区",
                "detail": "南京路88号",
                "set_default": True,
                "impact_summary": "沿用当前默认联系人，创建郑州地址并设为默认地址",
            },
        }]),
    ])
    monkeypatch.setattr(tool_agent, "invoke_llm", lambda *_args, **_kwargs: next(responses))
    assignment = AgentAssignment(
        task_id="R-nested-clarify",
        agent="retail_agent",
        objective="将默认地址改为河南省郑州市二七区南京路88号",
        capability="default_address",
        arguments={
            "address_line": "河南省郑州市二七区南京路88号",
            "contact_strategy": "reuse_current_default",
        },
    )

    result = asyncio.run(tool_agent.run_tool_agent(
        assignment,
        create_chat_state("u1", "s-nested-clarify", assignment.objective),
        _identity("s-nested-clarify"),
    ))

    assert result.status == "needs_confirmation"
    assert result.error_code is None
    completed = asyncio.run(actions.confirm(_identity("s-nested-clarify")))
    assert completed.status == "succeeded"
    default_address = actions.business.get_default_address("u1")["address"]
    assert default_address["recipient"] == "苏军"
    assert default_address["phone"] == "15588697856"
    assert default_address["province"] == "河南省"
    assert default_address["detail"] == "南京路88号"
    assert store.get_owned("addresses", "A1", "u1")["is_default"] is False


def test_natural_language_confirmation_is_applied_only_after_llm_decision(monkeypatch) -> None:
    store, actions = _platform()
    actions.propose_write(
        "telecom_change_plan",
        {"line_id": "L1", "plan_id": "P2", "expected_version": 1},
        _identity(),
        impact_summary="将线路 L1 变更为畅享套餐",
    )
    calls = 0

    def invoke(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return Response(
            '{"action":"confirm_action","standalone_query":"同意执行待确认套餐变更",'
            '"assignments":[],"confidence":0.99}'
        )

    monkeypatch.setattr(supervisor_agent, "invoke_llm", invoke)
    state = create_chat_state("u1", "s1", "行，就按刚才说的办")

    state.update(asyncio.run(supervisor_graph_nodes.supervisor_manager_node(state)))
    assert supervisor_graph_nodes.supervisor_route(state) == "action"
    state.update(asyncio.run(supervisor_graph_nodes.pending_action_execution_node(state)))

    assert calls == 1
    assert state["intent"] == "action_confirmation"
    assert store.get_owned("lines", "L1", "u1")["current_plan_id"] == "P2"


def test_exact_confirmation_bypasses_supervisor_llm(monkeypatch) -> None:
    store, actions = _platform()
    actions.propose_write(
        "telecom_change_plan",
        {"line_id": "L1", "plan_id": "P2", "expected_version": 1},
        _identity(),
        impact_summary="将线路 L1 变更为畅享套餐",
    )
    monkeypatch.setattr(
        supervisor_agent,
        "invoke_llm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("exact confirmation must not call the LLM")
        ),
    )
    state = create_chat_state("u1", "s1", "确认")

    update = asyncio.run(supervisor_graph_nodes.supervisor_manager_node(state))
    state.update(update)

    assert update["supervisor_decision"]["action"] == "confirm_action"
    assert update["supervisor_decision"]["confidence"] == 1.0
    assert update["node_logs"] == [
        "Supervisor decision: confirm_action (deterministic_action_command)"
    ]
    assert supervisor_graph_nodes.supervisor_route(state) == "action"
    state.update(asyncio.run(supervisor_graph_nodes.pending_action_execution_node(state)))
    assert state["intent"] == "action_confirmation"
    assert store.get_owned("lines", "L1", "u1")["current_plan_id"] == "P2"


def test_exact_cancellation_with_punctuation_bypasses_supervisor_llm(monkeypatch) -> None:
    store, actions = _platform()
    actions.propose_write(
        "telecom_change_plan",
        {"line_id": "L1", "plan_id": "P2", "expected_version": 1},
        _identity(),
        impact_summary="将线路 L1 变更为畅享套餐",
    )
    monkeypatch.setattr(
        supervisor_agent,
        "invoke_llm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("exact cancellation must not call the LLM")
        ),
    )
    state = create_chat_state("u1", "s1", "取消！")

    state.update(asyncio.run(supervisor_graph_nodes.supervisor_manager_node(state)))

    assert state["supervisor_decision"]["action"] == "reject_action"
    assert supervisor_graph_nodes.supervisor_route(state) == "action"
    state.update(asyncio.run(supervisor_graph_nodes.pending_action_execution_node(state)))
    assert state["intent"] == "action_rejection"
    assert "。。" not in state["sub_results"]["supervisor"]
    assert store.get_owned("lines", "L1", "u1")["current_plan_id"] == "P1"


def test_multiple_write_assignments_are_queued_and_only_first_is_released(monkeypatch) -> None:
    _platform()

    async def decide(*_args, **_kwargs):
        return SupervisorDecision(
            action="dispatch",
            standalone_query="变更套餐后开启漫游",
            assignments=(
                AgentAssignment(
                    task_id="W1",
                    agent="telecom_agent",
                    objective="将线路 L1 变更为套餐 P2",
                    capability="plan_change",
                    arguments={"line_id": "L1", "plan_id": "P2"},
                ),
                AgentAssignment(
                    task_id="W2",
                    agent="telecom_agent",
                    objective="为线路 L1 开启漫游",
                    capability="roaming",
                    arguments={"line_id": "L1", "enabled": True},
                ),
            ),
            confidence=0.99,
        )

    monkeypatch.setattr(supervisor_graph_nodes, "decide_next_step", decide)
    state = create_chat_state("u1", "multi-write", "变更套餐后开启漫游", turn_id="source-turn")

    update = asyncio.run(supervisor_graph_nodes.supervisor_manager_node(state))
    plan = PendingWritePlan.model_validate(update["pending_write_plan"])

    assert [item["task_id"] for item in update["agent_assignments"]] == ["W1"]
    assert [item.task_id for item in plan.assignments] == ["W1", "W2"]
    assert plan.current_index == 0
    assert plan.active_action_id is None


def test_successful_confirmation_revalidates_and_proposes_next_queued_write(monkeypatch) -> None:
    store, actions = _platform()
    identity = _identity("multi-write")
    first_action = actions.propose_write(
        "telecom_change_plan",
        {"line_id": "L1", "plan_id": "P2", "expected_version": 1},
        identity,
        impact_summary="将线路 L1 变更为套餐 P2",
    )
    plan = PendingWritePlan(
        plan_id="plan-1",
        source_turn_id="source-turn",
        assignments=(
            AgentAssignment(
                task_id="W1",
                agent="telecom_agent",
                objective="将线路 L1 变更为套餐 P2",
                capability="plan_change",
            ),
            AgentAssignment(
                task_id="W2",
                agent="telecom_agent",
                objective="为线路 L1 开启漫游",
                capability="roaming",
                arguments={"line_id": "L1", "enabled": True},
            ),
        ),
        active_action_id=first_action.action_id,
    )
    state = create_chat_state(
        "u1",
        "multi-write",
        "确认",
        turn_id="confirmation-turn-1",
        pending_write_plan=plan.model_dump(mode="json"),
    )

    state.update(asyncio.run(supervisor_graph_nodes.supervisor_manager_node(state)))
    state.update(asyncio.run(supervisor_graph_nodes.pending_action_execution_node(state)))
    assert store.get_owned("lines", "L1", "u1")["current_plan_id"] == "P2"
    assert state["continue_write_plan"] is True
    assert state["agent_assignments"][0]["task_id"] == "W2"
    advanced = PendingWritePlan.model_validate(state["pending_write_plan"])
    assert advanced.current_index == 1
    assert advanced.active_action_id is None

    async def propose_next(assignment, _state, current_identity):
        # The first action incremented the line version.  The next proposal is
        # therefore built from a fresh version rather than the original plan.
        current = store.get_owned("lines", "L1", "u1")
        assert current["version"] == 2
        next_action = actions.propose_write(
            "telecom_set_roaming",
            {
                "line_id": "L1",
                "enabled": True,
                "expected_version": current["version"],
            },
            current_identity,
            impact_summary="为线路 L1 开启漫游",
        )
        return AgentResult(
            task_id=assignment.task_id,
            agent=assignment.agent,
            status="needs_confirmation",
            pending_action_id=next_action.action_id,
            user_fragment="待确认：为线路 L1 开启漫游。",
        )

    monkeypatch.setattr(supervisor_graph_nodes, "_execute_assignment", propose_next)
    state.update(asyncio.run(supervisor_graph_nodes.domain_dispatch_node(state)))
    rebound = PendingWritePlan.model_validate(state["pending_write_plan"])

    assert rebound.active_action_id
    assert "操作已完成" in state["supervisor_response"]
    assert "开启漫游" in state["supervisor_response"]
    assert store.get_owned("lines", "L1", "u1")["roaming_enabled"] is False

    final_state = create_chat_state(
        "u1",
        "multi-write",
        "确认",
        turn_id="confirmation-turn-2",
        pending_write_plan=rebound.model_dump(mode="json"),
    )
    final_state.update(asyncio.run(supervisor_graph_nodes.supervisor_manager_node(final_state)))
    final_state.update(asyncio.run(supervisor_graph_nodes.pending_action_execution_node(final_state)))

    assert store.get_owned("lines", "L1", "u1")["roaming_enabled"] is True
    assert final_state["pending_write_plan"] is None
    assert final_state["continue_write_plan"] is False


def test_rejecting_current_write_stops_remaining_plan() -> None:
    store, actions = _platform()
    first_action = actions.propose_write(
        "telecom_change_plan",
        {"line_id": "L1", "plan_id": "P2", "expected_version": 1},
        _identity("multi-write-reject"),
        impact_summary="将线路 L1 变更为套餐 P2",
    )
    plan = PendingWritePlan(
        plan_id="plan-reject",
        source_turn_id="source-turn",
        assignments=(
            AgentAssignment(task_id="W1", agent="telecom_agent", objective="变更套餐", capability="plan_change"),
            AgentAssignment(task_id="W2", agent="telecom_agent", objective="开启漫游", capability="roaming"),
        ),
        active_action_id=first_action.action_id,
    )
    state = create_chat_state(
        "u1",
        "multi-write-reject",
        "取消",
        turn_id="reject-turn",
        pending_write_plan=plan.model_dump(mode="json"),
    )

    state.update(asyncio.run(supervisor_graph_nodes.supervisor_manager_node(state)))
    state.update(asyncio.run(supervisor_graph_nodes.pending_action_execution_node(state)))

    assert state["pending_write_plan"] is None
    assert state["continue_write_plan"] is False
    assert "剩余 1 项写操作已停止" in state["supervisor_response"]
    assert store.get_owned("lines", "L1", "u1")["current_plan_id"] == "P1"


def test_same_request_confirmation_cannot_approve_newly_created_next_action(monkeypatch) -> None:
    _store, actions = _platform()
    actions.propose_write(
        "telecom_set_roaming",
        {"line_id": "L1", "enabled": True, "expected_version": 1},
        RequestIdentityContext(user_id="u1", session_id="retry-guard", turn_id="same-turn"),
        impact_summary="为线路 L1 开启漫游",
    )
    monkeypatch.setattr(
        supervisor_agent,
        "invoke_llm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("same-turn confirmation must be blocked deterministically")
        ),
    )
    state = create_chat_state(
        "u1",
        "retry-guard",
        "确认",
        turn_id="same-turn",
    )

    update = asyncio.run(supervisor_graph_nodes.supervisor_manager_node(state))

    assert update["supervisor_decision"]["action"] == "clarify"
    assert "新的消息" in update["supervisor_response"]


def test_exact_cancel_stops_write_plan_waiting_for_clarification(monkeypatch) -> None:
    _platform()
    monkeypatch.setattr(
        supervisor_agent,
        "invoke_llm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("exact plan cancellation must be deterministic")
        ),
    )
    plan = PendingWritePlan(
        plan_id="plan-clarify",
        source_turn_id="source-turn",
        assignments=(
            AgentAssignment(task_id="W1", agent="telecom_agent", objective="变更套餐", capability="plan_change"),
            AgentAssignment(task_id="W2", agent="telecom_agent", objective="开启漫游", capability="roaming"),
        ),
    )
    state = create_chat_state(
        "u1",
        "clarify-cancel",
        "取消",
        turn_id="cancel-turn",
        pending_write_plan=plan.model_dump(mode="json"),
    )

    update = asyncio.run(supervisor_graph_nodes.supervisor_manager_node(state))

    assert update["pending_write_plan"] is None
    assert update["pending_task"] is None
    assert update["supervisor_decision"]["action"] == "finish"
    assert "2 项操作未执行" in update["supervisor_response"]


def test_knowledge_agent_owns_domain_rag(monkeypatch) -> None:
    async def retrieve(_state, domain, capability):
        assert domain == "retail"
        assert capability == "retail_policy"
        return {
            "sub_results": {"supervisor": "退货政策答案[1]"},
            "task_results": {"rag": {"grounded": True, "citations": [{"source": "policy.md"}]}},
        }

    monkeypatch.setattr(knowledge_agent, "retrieve_grounded_answer", retrieve)
    assignment = AgentAssignment(
        task_id="K1",
        agent="knowledge_agent",
        objective="查询退货期限政策",
        capability="retail_policy",
    )

    result = asyncio.run(knowledge_agent.run_knowledge_agent(
        assignment,
        create_chat_state("u1", "s-k", "退货期限是多久"),
    ))

    assert result.agent == "knowledge_agent"
    assert result.status == "succeeded"
    assert result.facts["rag"]["grounded"] is True


def test_tool_agent_receives_only_its_assigned_part_of_a_composite_request(monkeypatch) -> None:
    _platform()
    captured_messages = []
    captured_tools = []
    responses = iter([
        AIMessage(content="", tool_calls=[{
            "id": "call-usage-1",
            "name": "telecom_get_usage_profile",
            "args": {},
        }]),
        Response('{"action":"final","response":"Current-month data usage was retrieved."}'),
    ])

    def invoke(messages, **kwargs):
        captured_messages.append(messages)
        captured_tools.append(kwargs["tools"])
        return next(responses)

    monkeypatch.setattr(tool_agent, "invoke_llm", invoke)
    objective = "check the current-month data usage for the authenticated user"
    assignment = AgentAssignment(
        task_id="T-usage",
        agent="telecom_agent",
        objective=objective,
        capability="usage",
    )

    state = create_chat_state(
        "u1",
        "s-composite",
        "check current data usage and whether I bought products this month",
    )
    result = asyncio.run(tool_agent.run_tool_agent(
        assignment,
        state,
        _identity("s-composite"),
    ))

    first_payload = json.loads(captured_messages[0][-1].content)
    assert len(captured_messages[0]) == 2
    assert isinstance(captured_messages[0][0], SystemMessage)
    assert isinstance(captured_messages[0][1], HumanMessage)
    assert first_payload["type"] == "agent_assignment"
    assert first_payload["user_query"] == objective
    assert "bought products" not in first_payload["user_query"]
    assert first_payload["conversation_context"]["summary"] == ""
    assert first_payload["conversation_context"]["recent_messages"] == []
    assert "observations" not in first_payload
    assert "若工具 Schema 未把 line_id" in captured_messages[0][0].content
    second_messages = captured_messages[1]
    assert len(second_messages) == 4
    assert isinstance(second_messages[2], AIMessage)
    assert isinstance(second_messages[3], ToolMessage)
    assert second_messages[2].tool_calls[0]["id"] == "call-usage-1"
    assert second_messages[3].tool_call_id == "call-usage-1"
    observation_payload = json.loads(second_messages[3].content)
    assert observation_payload["type"] == "tool_result"
    assert observation_payload["completed_step"] == 1
    assert observation_payload["observation"]["tool_name"] == "telecom_get_usage_profile"
    native_names = {
        item["function"]["name"]
        for item in captured_tools[0]
    }
    assert "telecom_get_usage_profile" in native_names
    assert result.status == "succeeded"
    observation = result.facts["observations"][0]
    assert observation["arguments"] == {}
    assert observation["result"]["line_id"] == "L1"
    assert len(state["messages"]) == 1
    assert state["messages"][0].content == (
        "check current data usage and whether I bought products this month"
    )
    assert "subagent_messages" not in state


def test_tool_agent_receives_only_explicit_dependency_results(monkeypatch) -> None:
    _platform()
    captured = {}
    responses = iter([
        Response(
            '{"action":"tool_call","tool_name":"telecom_get_current_plan",'
            '"arguments":{"line_id":"L1"}}'
        ),
        Response('{"action":"final","response":"已结合上游结果检查套餐。"}'),
    ])

    def invoke(messages, **_kwargs):
        if "payload" not in captured:
            captured["payload"] = json.loads(messages[-1].content)
            captured["system"] = messages[0].content
        return next(responses)

    monkeypatch.setattr(tool_agent, "invoke_llm", invoke)
    assignment = AgentAssignment(
        task_id="R-dependent",
        agent="telecom_agent",
        objective="结合上游知识检查当前套餐",
        capability="current_plan",
        dependencies=("K-policy",),
        arguments={"line_id": "L1"},
    )
    dependency = AgentResult(
        task_id="K-policy",
        agent="knowledge_agent",
        status="succeeded",
        facts={"rag": {"grounded": True}},
        user_fragment="订单送达后可按政策申请退货。[1]",
    ).model_dump(mode="json")

    result = asyncio.run(tool_agent.run_tool_agent(
        assignment,
        create_chat_state("u1", "s-dependent", assignment.objective),
        _identity("s-dependent"),
        dependency_results={"K-policy": dependency},
    ))

    assert result.status == "succeeded"
    assert captured["payload"]["dependency_results"] == {"K-policy": dependency}
    assert "不得替代写操作所需的本轮实时读取" in captured["system"]


def test_tool_agent_message_trajectory_does_not_cross_turn_invocations(monkeypatch) -> None:
    _platform()
    captured_messages = []
    responses = iter([
        AIMessage(content="", tool_calls=[{
            "id": "call-first-turn",
            "name": "telecom_get_usage_profile",
            "args": {},
        }]),
        Response('{"action":"final","response":"first turn usage"}'),
        AIMessage(content="", tool_calls=[{
            "id": "call-second-turn",
            "name": "telecom_get_usage_profile",
            "args": {},
        }]),
        Response('{"action":"final","response":"second turn usage"}'),
    ])

    def invoke(messages, **_kwargs):
        captured_messages.append(messages)
        return next(responses)

    monkeypatch.setattr(tool_agent, "invoke_llm", invoke)
    first_assignment = AgentAssignment(
        task_id="T-first",
        agent="telecom_agent",
        objective="FIRST TURN: check current usage",
        capability="usage",
    )
    second_assignment = AgentAssignment(
        task_id="T-second",
        agent="telecom_agent",
        objective="SECOND TURN: check current usage again",
        capability="usage",
    )

    first_result = asyncio.run(tool_agent.run_tool_agent(
        first_assignment,
        create_chat_state("u1", "same-session", first_assignment.objective, turn_id="turn-1"),
        _identity("same-session"),
    ))
    second_result = asyncio.run(tool_agent.run_tool_agent(
        second_assignment,
        create_chat_state("u1", "same-session", second_assignment.objective, turn_id="turn-2"),
        _identity("same-session"),
    ))

    assert first_result.status == "succeeded"
    assert second_result.status == "succeeded"
    assert len(captured_messages[1]) == 4
    assert isinstance(captured_messages[1][3], ToolMessage)
    assert captured_messages[1][3].tool_call_id == "call-first-turn"
    assert len(captured_messages[2]) == 2
    second_payload = json.loads(captured_messages[2][1].content)
    assert second_payload["user_query"] == second_assignment.objective
    assert "FIRST TURN" not in captured_messages[2][1].content


def test_legacy_tool_json_is_normalized_to_tool_messages_during_migration(monkeypatch) -> None:
    _platform()
    captured_messages = []
    responses = iter([
        Response(
            '{"action":"tool_call","tool_name":"telecom_get_usage_profile",'
            '"arguments":{}}'
        ),
        Response('{"action":"final","response":"usage loaded"}'),
    ])

    def invoke(messages, **_kwargs):
        captured_messages.append(messages)
        return next(responses)

    monkeypatch.setattr(tool_agent, "invoke_llm", invoke)
    assignment = AgentAssignment(
        task_id="T-legacy-tool-call",
        agent="telecom_agent",
        objective="check current usage",
        capability="usage",
    )

    result = asyncio.run(tool_agent.run_tool_agent(
        assignment,
        create_chat_state("u1", "legacy-tool-call", assignment.objective),
        _identity("legacy-tool-call"),
    ))

    assert result.status == "succeeded"
    replay = captured_messages[1]
    assert isinstance(replay[2], AIMessage)
    assert replay[2].tool_calls[0]["id"] == "legacy-tool-call-1-1"
    assert isinstance(replay[3], ToolMessage)
    assert replay[3].tool_call_id == "legacy-tool-call-1-1"


def test_clarification_result_preserves_structured_assignment_for_next_turn(monkeypatch) -> None:
    _platform()

    async def clarify(assignment, _state, _identity_context):
        return AgentResult(
            task_id=assignment.task_id,
            agent=assignment.agent,
            status="needs_clarification",
            facts={"missing_fields": ["contact_strategy"]},
            user_fragment="请确认是否沿用默认地址联系人。",
        )

    monkeypatch.setattr(supervisor_graph_nodes, "_execute_assignment", clarify)
    state = create_chat_state("u1", "pending-session", "修改默认地址")
    state["agent_assignments"] = [{
        "task_id": "R-pending",
        "agent": "retail_agent",
        "objective": "创建北京市朝阳区应天路88号并设为默认地址",
        "capability": "default_address",
        "dependencies": [],
        "arguments": {
            "province": "北京市",
            "city": "北京市",
            "district": "朝阳区",
            "detail": "应天路88号",
            "set_default": True,
        },
    }]

    update = asyncio.run(supervisor_graph_nodes.domain_dispatch_node(state))

    assert update["pending_task"]["objective"] == "创建北京市朝阳区应天路88号并设为默认地址"
    assert update["pending_task"]["arguments"]["detail"] == "应天路88号"


def test_unrelated_assignment_does_not_clear_pending_task(monkeypatch) -> None:
    _platform()

    async def succeed(assignment, _state, _identity_context):
        return AgentResult(
            task_id=assignment.task_id,
            agent=assignment.agent,
            status="succeeded",
            user_fragment="查询完成",
        )

    monkeypatch.setattr(supervisor_graph_nodes, "_execute_assignment", succeed)
    state = create_chat_state(
        "u1",
        "pending-unrelated",
        "查询套餐",
        pending_task={
            "agent": "retail_agent",
            "capability": "default_address",
            "objective": "创建北京默认地址",
            "arguments": {},
        },
    )
    state["agent_assignments"] = [{
        "task_id": "T-unrelated",
        "agent": "telecom_agent",
        "objective": "查询当前套餐",
        "capability": "current_plan",
        "dependencies": [],
        "arguments": {},
    }]

    update = asyncio.run(supervisor_graph_nodes.domain_dispatch_node(state))

    assert "pending_task" not in update


def test_round_limit_rejects_further_dispatch_and_uses_completed_result(monkeypatch) -> None:
    _platform()
    monkeypatch.setattr(
        supervisor_agent,
        "invoke_llm",
        lambda *_args, **_kwargs: Response(
            '{"action":"dispatch","standalone_query":"继续查询",'
            '"assignments":[{"task_id":"T2","agent":"telecom_agent",'
            '"objective":"再次查询套餐","capability":"current_plan",'
            '"dependencies":[],"arguments":{}}],"confidence":0.99}'
        ),
    )
    state = create_chat_state("u1", "round-limit", "继续")
    state["supervisor_round"] = 3
    state["task_results"] = {
        "T1": AgentResult(
            task_id="T1",
            agent="telecom_agent",
            status="succeeded",
            user_fragment="已有可靠结果",
        ).model_dump(mode="json")
    }

    update = asyncio.run(supervisor_graph_nodes.supervisor_manager_node(state))

    assert update["supervisor_decision"]["action"] == "finish"
    assert update["agent_assignments"] == []
    assert update["supervisor_response"] == "已有可靠结果"
