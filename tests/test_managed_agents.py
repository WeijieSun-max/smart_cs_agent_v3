from __future__ import annotations

import asyncio
import json

from domain.action_governance import GovernedActionService, initialize_action_service
from domain.business.service import initialize_service
from domain.business.store import InMemoryBusinessStore
from domain.customer_service_agent.agents import knowledge_agent, supervisor_agent, tool_agent
from domain.customer_service_agent.orchestration.models import AgentAssignment, AgentResult
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
    responses = iter([
        Response(
            '{"action":"propose_write","tool_name":"retail_create_address",'
            '"arguments":{"province":"北京市","city":"北京市","district":"朝阳区",'
            '"detail":"应天路88号","set_default":true},'
            '"impact_summary":"沿用当前默认联系人，创建北京地址并设为默认地址"}'
        ),
    ])

    def invoke(messages, **_kwargs):
        captured_messages.append(messages)
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
    first_payload = json.loads(captured_messages[0][-1].content)
    assert first_payload["observations"][0]["tool_name"] == "retail_get_default_address"
    assert first_payload["observations"][0]["result"]["address"]["recipient"] == "苏军"
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


def test_natural_language_confirmation_is_applied_only_after_llm_decision(monkeypatch) -> None:
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
        lambda *_args, **_kwargs: Response(
            '{"action":"confirm_action","standalone_query":"同意执行待确认套餐变更",'
            '"assignments":[],"confidence":0.99}'
        ),
    )
    state = create_chat_state("u1", "s1", "行，就按刚才说的办")

    state.update(asyncio.run(supervisor_graph_nodes.supervisor_manager_node(state)))
    assert supervisor_graph_nodes.supervisor_route(state) == "action"
    state.update(asyncio.run(supervisor_graph_nodes.pending_action_execution_node(state)))

    assert state["intent"] == "action_confirmation"
    assert store.get_owned("lines", "L1", "u1")["current_plan_id"] == "P2"


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
    responses = iter([
        Response('{"action":"tool_call","tool_name":"telecom_get_usage_profile","arguments":{}}'),
        Response('{"action":"final","response":"Current-month data usage was retrieved."}'),
    ])

    def invoke(messages, **_kwargs):
        captured_messages.append(messages)
        return next(responses)

    monkeypatch.setattr(tool_agent, "invoke_llm", invoke)
    objective = "check the current-month data usage for the authenticated user"
    assignment = AgentAssignment(
        task_id="T-usage",
        agent="telecom_agent",
        objective=objective,
        capability="usage",
    )

    result = asyncio.run(tool_agent.run_tool_agent(
        assignment,
        create_chat_state(
            "u1",
            "s-composite",
            "check current data usage and whether I bought products this month",
        ),
        _identity("s-composite"),
    ))

    first_payload = json.loads(captured_messages[0][-1].content)
    assert first_payload["user_query"] == objective
    assert "bought products" not in first_payload["user_query"]
    assert first_payload["conversation_context"]["summary"] == ""
    assert first_payload["conversation_context"]["recent_messages"] == []
    assert "若工具 Schema 未把 line_id" in captured_messages[0][0].content
    assert result.status == "succeeded"
    observation = result.facts["observations"][0]
    assert observation["arguments"] == {}
    assert observation["result"]["line_id"] == "L1"


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
