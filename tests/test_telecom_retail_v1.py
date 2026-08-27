from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from domain.action_governance import GovernedActionService,initialize_action_service
from domain.customer_service_agent.orchestration.models import TaskPlan, TaskSpec
from domain.shared.identity import RequestIdentityContext
from domain.shared.llm.llm_service import initialize_llm_client
from domain.business.service import initialize_service
from domain.business.store import InMemoryBusinessStore
from domain.customer_service_agent.file_skills.catalog import FileSkillCatalog,initialize_catalog
from domain.customer_service_agent.tools.tool_registry import get_mcp_server
from domain.customer_service_agent.workflow.entity.chat_state import create_chat_state
from domain.customer_service_agent.workflow.nodes.supervisor_node import supervisor_node


def fixtures():
    return {
        "users":[{"user_id":"u1","status":"active"},{"user_id":"u2","status":"active"}],
        "accounts":[{"account_id":"a1","user_id":"u1","status":"active"}],
        "lines":[{"line_id":"L1","account_id":"a1","status":"active","current_plan_id":"P1","version":1,"roaming_enabled":False}],
        "plans":[
            {"plan_id":"P1","name":"基础套餐","status":"active","monthly_price":"80","currency":"CNY","data_limit_mb":10240,"included_voice_minutes":100,"data_unlimited":False,"voice_unlimited":False,"voice_overage_price_per_minute":"0.1","refuel_price_per_gb":"10"},
            {"plan_id":"P2","name":"畅享套餐","status":"active","monthly_price":"90","currency":"CNY","data_limit_mb":30720,"included_voice_minutes":500,"data_unlimited":False,"voice_unlimited":False,"voice_overage_price_per_minute":"0.1","refuel_price_per_gb":"10"},
        ],
        "usage_cycles":[
            {"usage_id":"U1","line_id":"L1","cycle_start":"2026-05-01","used_data_mb":22000,"used_voice_minutes":300,"current":False,"refuel_count":1},
            {"usage_id":"U2","line_id":"L1","cycle_start":"2026-06-01","used_data_mb":24000,"used_voice_minutes":330,"current":False,"refuel_count":1},
            {"usage_id":"U3","line_id":"L1","cycle_start":"2026-07-01","used_data_mb":23000,"used_voice_minutes":310,"current":False,"refuel_count":0},
        ],
        "orders":[{"order_id":"O1","order_no":"NO1","user_id":"u1","status":"pending","grand_total":"100","currency":"CNY","version":1}],
        "products":[{"product_id":"PR1","name":"手机壳","description":"透明","status":"active"}],
    }


def setup_platform():
    store=InMemoryBusinessStore(fixtures()); business=initialize_service(store)
    actions=GovernedActionService(get_mcp_server(),business,ttl_seconds=900,tool_timeout_seconds=5)
    initialize_action_service(actions)
    return store,business,actions


def identity(user_id="u1"):
    return RequestIdentityContext(user_id=user_id,session_id="s1",turn_id="t1")


def test_file_skill_catalog_progressive_load_and_contract():
    catalog=FileSkillCatalog(Path(__file__).parents[1]/"skills",get_mcp_server()); catalog.scan_and_freeze()
    entry=catalog.select(capability="plan_recommendation",agent_type="telecom_agent")
    assert entry is not None
    loaded=catalog.load(entry.metadata.name,entry.metadata.version,agent_type="telecom_agent")
    assert "references/recommendation-policy.md" in loaded.references
    assert loaded.metadata.allowed_tools==("telecom_get_current_plan","telecom_get_usage_profile","telecom_list_plans","telecom_compare_plans")


def test_plan_comparison_is_deterministic_and_owned():
    _,business,_=setup_platform(); profile=business.usage_profile("u1","L1")
    assert profile.data_quality=="normal_confidence" and profile.recommended_data_mb>24000
    comparisons=business.compare_plans("u1","L1",["P1","P2"])
    assert comparisons[0]["plan_id"]=="P2"
    with pytest.raises(Exception): business.usage_profile("u2","L1")


def test_write_requires_frozen_confirmation_and_second_user_lookup():
    store,_,actions=setup_platform(); ctx=identity()
    action=actions.propose_write("telecom_change_plan",{"line_id":"L1","plan_id":"P2","expected_version":1},ctx,impact_summary="变更套餐")
    assert store.get_owned("lines","L1","u1")["current_plan_id"]=="P1"
    completed=asyncio.run(actions.confirm(ctx))
    assert completed.status=="succeeded"
    assert store.get_owned("lines","L1","u1")["current_plan_id"]=="P2"


def test_raw_write_tool_is_unreachable():
    setup_platform()
    result=asyncio.run(get_mcp_server().call_tool("retail_cancel_order",{"order_id":"O1","expected_version":1,"reason":"changed_mind"},trusted_context={"user_id":"u1"}))
    assert not result.success and result.error_code=="tool.validation"


def test_retail_cancel_is_owned_versioned_and_confirmed():
    store,_,actions=setup_platform(); ctx=identity()
    action=actions.propose_write("retail_cancel_order",{"order_id":"O1","expected_version":1,"reason":"changed_mind"},ctx,impact_summary="取消订单")
    assert action.status=="awaiting_confirmation"
    completed=asyncio.run(actions.confirm(ctx))
    assert completed.receipt["version_after"]==2
    assert store.get_owned("orders","O1","u1")["status"]=="cancelled"


def test_task_plan_rejects_cycles():
    plan=TaskPlan(tasks=(TaskSpec(task_id="T1",domain="telecom",capability="usage",dependencies=("T2",)),TaskSpec(task_id="T2",domain="retail",capability="order_query",dependencies=("T1",))))
    with pytest.raises(ValueError,match="cycle"): plan.validate_dag()


def test_supervisor_runs_skill_md_plan_recommendation_end_to_end():
    setup_platform(); initialize_catalog(Path(__file__).parents[1]/"skills",get_mcp_server())
    initialize_llm_client(FakeListChatModel(responses=[
        '{"action":"dispatch","standalone_query":"结合最近流量和通话推荐套餐",'
        '"assignments":[{"task_id":"T1","agent":"telecom_agent",'
        '"objective":"结合最近三期使用画像比较可用套餐并给出只读建议",'
        '"capability":"plan_recommendation","dependencies":[],"arguments":{"line_id":"L1"}}],'
        '"confidence":0.99}',
        '{"action":"tool_call","tool_name":"telecom_get_current_plan","arguments":{"line_id":"L1"}}',
        '{"action":"tool_call","tool_name":"telecom_get_usage_profile","arguments":{"line_id":"L1"}}',
        '{"action":"tool_call","tool_name":"telecom_list_plans","arguments":{"line_id":"L1"}}',
        '{"action":"tool_call","tool_name":"telecom_compare_plans",'
        '"arguments":{"line_id":"L1","candidate_plan_ids":["P1","P2"]}}',
        '{"action":"final","response":"主推荐 P2；这是只读建议，尚未变更套餐。"}',
        '{"action":"finish","standalone_query":"结合最近流量和通话推荐套餐",'
        '"response":"主推荐 P2；这是只读建议，尚未变更套餐。","confidence":0.99}',
    ]))
    state=create_chat_state("u1","s-plan","结合最近流量和通话帮我推荐套餐")
    result=asyncio.run(supervisor_node(state))
    assert result["skill_selection"]["skill_name"]=="telecom-plan-recommendation"
    assert "P2" in str(result["skill_result"]["facts"]["observations"])
    assert "尚未变更套餐" in result["sub_results"]["supervisor"]
