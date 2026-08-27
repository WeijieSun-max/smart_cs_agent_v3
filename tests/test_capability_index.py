from domain.customer_service_agent.orchestration.capability_index import (
    get_capability_index,
    validate_capability_contracts,
)


def test_write_capabilities_are_derived_from_registry() -> None:
    index = get_capability_index()

    assert "cancel_order" in index.write_capabilities
    assert "request_return" in index.write_capabilities
    # 写意图的能力，每个都有对应 write 工具。
    for cap in index.write_capabilities:
        assert cap in index.write_tools_by_capability


def test_agent_read_allowlist_excludes_write_intent_quote_tools() -> None:
    index = get_capability_index()

    telecom = index.allowed_read_tools("telecom", "telecom_agent")
    retail = index.allowed_read_tools("retail", "retail_agent")

    # 只读报价/预检工具（capability 是写能力）不得进入 ReAct 白名单。
    assert "telecom_quote_plan_change" not in telecom
    assert "telecom_quote_refuel" not in telecom
    # 核心只读工具保留。
    assert {"telecom_get_current_plan", "telecom_get_usage_profile", "telecom_list_plans", "telecom_compare_plans"} <= telecom
    # 零售侧含订单明细等只读工具。
    assert {"retail_get_order", "retail_get_order_detail", "retail_find_orders", "retail_list_products"} <= retail


def test_capability_contracts_are_consistent() -> None:
    assert validate_capability_contracts() == []
