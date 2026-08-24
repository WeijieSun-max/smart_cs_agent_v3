from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from domain.action_governance import get_action_service
from domain.customer_service_agent.file_skills import get_catalog
from domain.customer_service_agent.orchestration.planner import extract_entities
from domain.customer_service_agent.orchestration.state_updates import build_supervisor_result
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.identity import RequestIdentityContext


async def execute_plan_recommendation(
    state: ChatState,
    identity: RequestIdentityContext,
) -> dict[str, Any]:
    catalog = get_catalog()
    entry = catalog.select(capability="plan_recommendation", agent_type="telecom_agent")
    if entry is None:
        return build_supervisor_result(state, "telecom", "套餐推荐能力暂不可用。")
    skill = catalog.load(
        entry.metadata.name,
        entry.metadata.version,
        agent_type="telecom_agent",
    )
    line_id = extract_entities(state["raw_query"]).get("line_id")
    arguments = {"line_id": line_id} if line_id else {}
    actions = get_action_service()
    current, usage, plans = await asyncio.gather(
        actions.execute_read("telecom_get_current_plan", arguments, identity, skill=skill),
        actions.execute_read("telecom_get_usage_profile", arguments, identity, skill=skill),
        actions.execute_read("telecom_list_plans", arguments, identity, skill=skill),
    )
    comparisons = await actions.execute_read(
        "telecom_compare_plans",
        {
            "line_id": current["line_id"],
            "candidate_plan_ids": [item["plan_id"] for item in plans],
        },
        identity,
        skill=skill,
    )
    text = _recommendation_text(current, usage, comparisons)
    selection = {
        **skill.identity(),
        "capability": "plan_recommendation",
        "agent_type": "telecom_agent",
    }
    return {
        **build_supervisor_result(state, "telecom", text),
        "skill_selection": selection,
        "skill_result": {
            "status": "succeeded",
            "facts": {
                "current_plan": current,
                "usage_profile": usage,
                "comparisons": comparisons,
            },
            **selection,
        },
    }


def _recommendation_text(
    current: dict[str, Any],
    usage: dict[str, Any],
    comparisons: list[dict[str, Any]],
) -> str:
    viable = [
        item
        for item in comparisons
        if item["eligible"]
        and not item["dominated"]
        and item["meets_data_need"]
        and item["meets_voice_need"]
    ]
    primary = min(
        viable or comparisons,
        key=lambda item: Decimal(str(item["expected_monthly_cost"])),
    ) if comparisons else None
    if primary is None:
        return "目前没有可比较的有效套餐，无法给出可靠推荐。"
    if usage["data_quality"] == "insufficient":
        return (
            f"目前完整账期数据不足，暂不下确定性结论。当前套餐为 {current['name']}"
            f"（月租 {current['monthly_price']} {current.get('currency', 'CNY')}）。"
            "可继续观察一个完整账期后再比较。"
        )
    current_comparison = next(
        (item for item in comparisons if item["plan_id"] == current["plan_id"]),
        None,
    )
    keep = bool(
        current_comparison
        and (
            primary["plan_id"] == current["plan_id"]
            or Decimal(str(primary["expected_monthly_cost"]))
            > Decimal(str(current_comparison["expected_monthly_cost"])) * Decimal("0.90")
        )
    )
    chosen = current_comparison if keep and current_comparison else primary
    reason = (
        f"预计数据需求约 {usage['recommended_data_mb']}MB，语音约 "
        f"{usage['recommended_voice_minutes']}分钟；该套餐预计月总成本 "
        f"{chosen['expected_monthly_cost']}。"
    )
    prefix = (
        f"建议保持当前套餐 {current['name']}。"
        if keep
        else f"主推荐：{chosen['name']}（月租 {chosen['monthly_price']} {current.get('currency', 'CNY')}）。"
    )
    return (
        prefix
        + reason
        + f"\n分析范围：{usage['analysis_period']}，数据质量：{usage['data_quality']}。"
        + "\n这是只读建议，尚未变更套餐；真正办理前会重新报价并要求二次确认。"
    )
