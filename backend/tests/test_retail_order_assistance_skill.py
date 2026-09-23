from __future__ import annotations

import asyncio
import json
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from domain.action_governance import GovernedActionService, initialize_action_service
from domain.business.service import initialize_service
from domain.business.store import InMemoryBusinessStore
from domain.customer_service_agent.agents import tool_agent
from domain.customer_service_agent.file_skills.catalog import initialize_catalog
from domain.customer_service_agent.orchestration.models import AgentAssignment
from domain.customer_service_agent.tools.tool_registry import get_mcp_server
from domain.customer_service_agent.workflow.entity.chat_state import create_chat_state
from domain.shared.identity import RequestIdentityContext
from domain.shared.llm.llm_service import initialize_llm_client


class ToolBindingFakeListChatModel(FakeListChatModel):
    """LangChain deterministic fake with no-op native tool binding."""

    def bind_tools(self, _tools, **_kwargs):
        return self


SKILLS_ROOT = Path(__file__).parents[1] / "skills"
SKILL_ROOT = SKILLS_ROOT / "retail" / "order-assistance"


def _setup_platform() -> GovernedActionService:
    store = InMemoryBusinessStore({
        "users": [{"user_id": "u1", "status": "active"}],
        "orders": [
            {
                "order_id": "O1",
                "order_no": "NO-001",
                "user_id": "u1",
                "status": "pending",
                "grand_total": "599.00",
                "currency": "CNY",
                "version": 1,
                "placed_at": "2026-08-21T10:00:00+00:00",
            },
            {
                "order_id": "O2",
                "order_no": "NO-002",
                "user_id": "u1",
                "status": "delivered",
                "grand_total": "129.00",
                "currency": "CNY",
                "version": 1,
                "placed_at": "2026-08-12T10:00:00+00:00",
            },
        ],
        "order_items": [
            {
                "order_item_id": "I1",
                "order_id": "O1",
                "name_snapshot": "5G随身WiFi Pro",
                "sku_snapshot": "SKU-001",
                "quantity": 1,
            },
            {
                "order_item_id": "I2",
                "order_id": "O2",
                "name_snapshot": "Type-C快充套装",
                "sku_snapshot": "SKU-002",
                "quantity": 1,
            },
        ],
    })
    business = initialize_service(store)
    actions = GovernedActionService(get_mcp_server(), business)
    initialize_action_service(actions)
    return actions


def test_retail_order_skill_progressive_load_contract_and_evals() -> None:
    catalog = initialize_catalog(SKILLS_ROOT, get_mcp_server())

    entry = catalog.select(capability="order_query", agent_type="retail_agent")
    assert entry is not None
    loaded = catalog.load(entry.metadata.name, entry.metadata.version, agent_type="retail_agent")
    evals = json.loads((SKILL_ROOT / "evals" / "evals.json").read_text(encoding="utf-8"))

    assert loaded.metadata.name == "retail-order-assistance"
    assert loaded.metadata.allowed_tools == (
        "retail_find_orders",
        "retail_list_orders",
        "retail_get_order",
        "retail_get_order_detail",
    )
    assert "references/order-resolution-policy.md" in loaded.references
    assert "半开区间" in loaded.references["references/order-resolution-policy.md"]
    assert len(evals["evals"]) == 3
    assert catalog.select(capability="product_query", agent_type="retail_agent") is None
    assert catalog.select(capability="cancel_order", agent_type="retail_agent") is None
    assert catalog.select(capability="order_query", agent_type="telecom_agent") is None


def test_retail_agent_loads_skill_prompt_and_restricts_tool_whitelist(monkeypatch) -> None:
    _setup_platform()
    catalog = initialize_catalog(SKILLS_ROOT, get_mcp_server())
    captured_payloads: list[dict] = []
    responses = iter([
        '{"action":"tool_call","tool_name":"retail_find_orders","arguments":{}}',
        '{"action":"final","response":"最近有两笔订单：NO-001 和 NO-002。"}',
    ])

    def invoke(messages, **_kwargs):
        captured_payloads.append(json.loads(messages[1].content))
        return type("Response", (), {"content": next(responses)})()

    monkeypatch.setattr(tool_agent, "invoke_llm", invoke)
    result = asyncio.run(tool_agent.run_tool_agent(
        AgentAssignment(
            task_id="R1",
            agent="retail_agent",
            objective="查询最近购买的商品和订单",
            capability="order_query",
        ),
        create_chat_state("u1", "s1", "我最近有购买商品吗"),
        RequestIdentityContext(user_id="u1", session_id="s1", turn_id="t1"),
    ))

    available_names = {tool["name"] for tool in captured_payloads[0]["available_tools"]}
    skill_payload = captured_payloads[0]["skill"]
    assert available_names == set(catalog.select(
        capability="order_query",
        agent_type="retail_agent",
    ).metadata.allowed_tools)
    assert "retail_cancel_order" not in available_names
    assert "retail_list_products" not in available_names
    assert skill_payload["skill_name"] == "retail-order-assistance"
    assert "订单查找与消歧策略" in skill_payload["instructions"]
    assert "references/order-resolution-policy.md" in skill_payload["references"]
    assert result.status == "succeeded"
    assert result.facts["skill_selection"]["agent_type"] == "retail_agent"


def test_supervisor_runs_retail_order_skill_end_to_end(run_supervisor) -> None:
    _setup_platform()
    initialize_catalog(SKILLS_ROOT, get_mcp_server())
    initialize_llm_client(ToolBindingFakeListChatModel(responses=[
        '{"action":"dispatch","standalone_query":"查询最近购买的商品",'
        '"assignments":[{"task_id":"R1","agent":"retail_agent",'
        '"objective":"按时间倒序查询当前用户最近订单和商品",'
        '"capability":"order_query","dependencies":[],"arguments":{}}],'
        '"confidence":0.99}',
        '{"action":"tool_call","tool_name":"retail_find_orders","arguments":{}}',
        '{"action":"final","response":"最近有两笔订单：NO-001 和 NO-002。"}',
        '{"action":"finish","standalone_query":"查询最近购买的商品",'
        '"response":"最近有两笔订单：NO-001 和 NO-002。","confidence":0.99}',
    ]))

    result = asyncio.run(run_supervisor(
        create_chat_state("u1", "retail-skill", "我最近有购买商品吗")
    ))

    assert result["skill_selection"]["skill_name"] == "retail-order-assistance"
    observations = result["skill_result"]["facts"]["observations"]
    assert observations[0]["tool_name"] == "retail_find_orders"
    assert "NO-001" in result["sub_results"]["supervisor"]
