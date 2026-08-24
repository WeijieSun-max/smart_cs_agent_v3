from __future__ import annotations

import asyncio
from typing import Any

from domain.business import service as business_service
from pkg.exceptions.exception import ToolValidationError

from .tool_registry import server


def _user(context: dict[str, Any] | None) -> str:
    user_id = (context or {}).get("user_id")
    if not isinstance(user_id, str) or not user_id:
        raise ToolValidationError()
    return user_id


def _write_context(context: dict[str, Any] | None) -> tuple[str, str, str]:
    user_id = _user(context)
    action_id = (context or {}).get("action_id")
    idempotency_key = (context or {}).get("idempotency_key")
    if not isinstance(action_id, str) or not isinstance(idempotency_key, str):
        raise ToolValidationError()
    return user_id, action_id, idempotency_key


_LINE_SCHEMA = {"type": "object", "properties": {"line_id": {"type": "string", "minLength": 1, "maxLength": 26}}}


@server.register(name="telecom_get_current_plan", description="查询当前用户线路和当前套餐", input_schema=_LINE_SCHEMA, category="telecom", effect="read", supports_idempotency=False, domain="telecom", capabilities=("current_plan", "plan_recommendation"), allowed_agent_types=("telecom_agent",), parallel_safe=True)
async def telecom_get_current_plan(line_id: str | None = None, _trusted_context: dict | None = None) -> dict:
    return await asyncio.to_thread(business_service.get_service().current_plan, _user(_trusted_context), line_id)


@server.register(name="telecom_get_usage_profile", description="按近期账期聚合流量和语音通话使用画像", input_schema=_LINE_SCHEMA, category="telecom", effect="read", supports_idempotency=False, domain="telecom", capabilities=("usage", "plan_recommendation"), allowed_agent_types=("telecom_agent",), parallel_safe=True)
async def telecom_get_usage_profile(line_id: str | None = None, _trusted_context: dict | None = None) -> dict:
    profile = await asyncio.to_thread(business_service.get_service().usage_profile, _user(_trusted_context), line_id)
    return profile.model_dump(mode="json")


@server.register(name="telecom_list_plans", description="列出当前线路可选的有效套餐", input_schema=_LINE_SCHEMA, category="telecom", effect="read", supports_idempotency=False, domain="telecom", capabilities=("plan_catalog", "plan_recommendation"), allowed_agent_types=("telecom_agent",), parallel_safe=True)
async def telecom_list_plans(line_id: str | None = None, _trusted_context: dict | None = None) -> list[dict]:
    return await asyncio.to_thread(business_service.get_service().list_plans, _user(_trusted_context), line_id)


@server.register(name="telecom_compare_plans", description="确定性比较套餐容量和预计月成本", input_schema={"type":"object","properties":{"line_id":{"type":"string","minLength":1,"maxLength":26},"candidate_plan_ids":{"type":"array","items":{"type":"string","minLength":1,"maxLength":26},"minItems":1,"maxItems":20,"uniqueItems":True}},"required":["candidate_plan_ids"]}, category="telecom", effect="read", supports_idempotency=False, domain="telecom", capabilities=("plan_recommendation",), allowed_agent_types=("telecom_agent",), parallel_safe=True)
async def telecom_compare_plans(candidate_plan_ids: list[str], line_id: str | None = None, _trusted_context: dict | None = None) -> list[dict]:
    return await asyncio.to_thread(business_service.get_service().compare_plans, _user(_trusted_context), line_id, candidate_plan_ids)


@server.register(name="telecom_quote_plan_change", description="读取套餐变更前的最新线路版本和价格影响", input_schema={"type":"object","properties":{"line_id":{"type":"string","minLength":1,"maxLength":26},"plan_id":{"type":"string","minLength":1,"maxLength":26}},"required":["line_id","plan_id"]}, category="telecom", effect="read", supports_idempotency=False, domain="telecom", capabilities=("plan_change",), allowed_agent_types=("telecom_agent",), parallel_safe=False)
async def telecom_quote_plan_change(line_id: str, plan_id: str, _trusted_context: dict | None = None) -> dict:
    service=business_service.get_service(); user_id=_user(_trusted_context); current=await asyncio.to_thread(service.current_plan,user_id,line_id); plans=await asyncio.to_thread(service.list_plans,user_id,line_id); target=next((p for p in plans if p.get("plan_id")==plan_id),None)
    if target is None: raise ToolValidationError()
    return {"line_id":line_id,"plan_id":plan_id,"expected_version":current["line_version"],"current_plan":{"plan_id":current["plan_id"],"name":current["name"],"monthly_price":current["monthly_price"]},"target_plan":{"name":target["name"],"monthly_price":target["monthly_price"]}}


@server.register(name="telecom_quote_refuel", description="查询补充流量的确定价格和最新线路版本", input_schema={"type":"object","properties":{"line_id":{"type":"string","minLength":1,"maxLength":26},"amount_mb":{"type":"integer","minimum":1024,"maximum":102400,"multipleOf":1024}},"required":["line_id","amount_mb"]}, category="telecom", effect="read", supports_idempotency=False, domain="telecom", capabilities=("data_refuel",), allowed_agent_types=("telecom_agent",), parallel_safe=False)
async def telecom_quote_refuel(line_id: str, amount_mb: int, _trusted_context: dict | None = None) -> dict:
    current=await asyncio.to_thread(business_service.get_service().current_plan,_user(_trusted_context),line_id); price=float(current.get("refuel_price_per_gb",0))*amount_mb/1024
    return {"line_id":line_id,"amount_mb":amount_mb,"quoted_price":round(price,2),"currency":current.get("currency","CNY"),"expected_version":current["line_version"]}


async def _execute_write(tool_name: str, arguments: dict[str, Any], context: dict | None) -> dict:
    user_id, action_id, key = _write_context(context)
    return await asyncio.to_thread(business_service.get_service().execute_action, tool_name, arguments, user_id, action_id, key)


@server.register(name="telecom_change_plan", description="变更线路套餐；只能由确认后的治理动作执行", input_schema={"type":"object","properties":{"line_id":{"type":"string","minLength":1,"maxLength":26},"plan_id":{"type":"string","minLength":1,"maxLength":26},"expected_version":{"type":"integer","minimum":1}},"required":["line_id","plan_id","expected_version"]}, category="telecom", effect="write", supports_idempotency=True, domain="telecom", capabilities=("plan_change",), allowed_agent_types=("telecom_agent",), risk_level="medium")
async def telecom_change_plan(line_id: str, plan_id: str, expected_version: int, _trusted_context: dict | None = None) -> dict:
    return await _execute_write("telecom_change_plan",locals_without_context(locals()),_trusted_context)


@server.register(name="telecom_refuel_data", description="购买并补充流量；只能由确认后的治理动作执行", input_schema={"type":"object","properties":{"line_id":{"type":"string","minLength":1,"maxLength":26},"amount_mb":{"type":"integer","minimum":1024,"maximum":102400,"multipleOf":1024},"quoted_price":{"type":"number","minimum":0,"maximum":100000},"expected_version":{"type":"integer","minimum":1}},"required":["line_id","amount_mb","quoted_price","expected_version"]}, category="telecom", effect="write", supports_idempotency=True, domain="telecom", capabilities=("data_refuel",), allowed_agent_types=("telecom_agent",), risk_level="medium")
async def telecom_refuel_data(line_id: str, amount_mb: int, quoted_price: float, expected_version: int, _trusted_context: dict | None = None) -> dict:
    return await _execute_write("telecom_refuel_data",locals_without_context(locals()),_trusted_context)


@server.register(name="telecom_set_roaming", description="开启或关闭线路漫游；只能由确认后的治理动作执行", input_schema={"type":"object","properties":{"line_id":{"type":"string","minLength":1,"maxLength":26},"enabled":{"type":"boolean"},"expected_version":{"type":"integer","minimum":1}},"required":["line_id","enabled","expected_version"]}, category="telecom", effect="write", supports_idempotency=True, domain="telecom", capabilities=("roaming",), allowed_agent_types=("telecom_agent",), risk_level="medium")
async def telecom_set_roaming(line_id: str, enabled: bool, expected_version: int, _trusted_context: dict | None = None) -> dict:
    return await _execute_write("telecom_set_roaming",locals_without_context(locals()),_trusted_context)


@server.register(name="retail_list_products", description="查询在售商品", input_schema={"type":"object","properties":{"query":{"type":"string","maxLength":255}}}, category="retail", effect="read", supports_idempotency=False, domain="retail", capabilities=("product_query",), allowed_agent_types=("retail_agent",), parallel_safe=True)
async def retail_list_products(query: str = "", _trusted_context: dict | None = None) -> list[dict]:
    return await asyncio.to_thread(business_service.get_service().list_products,query)


@server.register(name="retail_get_order", description="查询属于当前用户的订单", input_schema={"type":"object","properties":{"order_id":{"type":"string","minLength":1,"maxLength":26}},"required":["order_id"]}, category="retail", effect="read", supports_idempotency=False, domain="retail", capabilities=("order_query",), allowed_agent_types=("retail_agent",), parallel_safe=True)
async def retail_get_order(order_id: str, _trusted_context: dict | None = None) -> dict:
    return await asyncio.to_thread(business_service.get_service().get_order,_user(_trusted_context),order_id)


@server.register(name="retail_list_orders", description="列出当前用户订单", input_schema={"type":"object","properties":{"status":{"type":"string","enum":["pending","processed","delivered","cancelled","return_requested","exchange_requested"]}}}, category="retail", effect="read", supports_idempotency=False, domain="retail", capabilities=("order_query",), allowed_agent_types=("retail_agent",), parallel_safe=True)
async def retail_list_orders(status: str | None = None, _trusted_context: dict | None = None) -> list[dict]:
    return await asyncio.to_thread(business_service.get_service().list_orders,_user(_trusted_context),status)


@server.register(name="retail_list_addresses",description="列出当前用户可用收货地址的脱敏信息",input_schema={"type":"object","properties":{}},category="retail",effect="read",supports_idempotency=False,domain="retail",capabilities=("address_query",),allowed_agent_types=("retail_agent",),parallel_safe=True)
async def retail_list_addresses(_trusted_context: dict | None=None) -> list[dict]:
    return await asyncio.to_thread(business_service.get_service().list_addresses,_user(_trusted_context))


@server.register(name="retail_list_payment_methods",description="列出当前用户可用支付方式的脱敏信息",input_schema={"type":"object","properties":{}},category="retail",effect="read",supports_idempotency=False,domain="retail",capabilities=("payment_query",),allowed_agent_types=("retail_agent",),parallel_safe=True)
async def retail_list_payment_methods(_trusted_context: dict | None=None) -> list[dict]:
    return await asyncio.to_thread(business_service.get_service().list_payment_methods,_user(_trusted_context))


_ORDER_WRITE_SCHEMAS: dict[str, dict[str, Any]] = {
    "retail_cancel_order":{"reason":{"type":"string","enum":["changed_mind","duplicate","wrong_item","other"]}},
    "retail_update_order_address":{"address_id":{"type":"string","minLength":1,"maxLength":26}},
    "retail_update_order_payment":{"payment_method_id":{"type":"string","minLength":1,"maxLength":26}},
    "retail_update_order_items":{"items":{"type":"array","items":{"type":"object","properties":{"variant_id":{"type":"string","minLength":1,"maxLength":26},"quantity":{"type":"integer","minimum":1,"maximum":99}},"required":["variant_id","quantity"],"additionalProperties":False},"minItems":1,"maxItems":50}},
    "retail_request_return":{"items":{"type":"array","items":{"type":"object","properties":{"order_item_id":{"type":"string","minLength":1,"maxLength":26},"quantity":{"type":"integer","minimum":1,"maximum":99}},"required":["order_item_id","quantity"],"additionalProperties":False},"minItems":1,"maxItems":50},"reason":{"type":"string","minLength":1,"maxLength":128}},
    "retail_request_exchange":{"items":{"type":"array","items":{"type":"object","properties":{"order_item_id":{"type":"string","minLength":1,"maxLength":26},"quantity":{"type":"integer","minimum":1,"maximum":99}},"required":["order_item_id","quantity"],"additionalProperties":False},"minItems":1,"maxItems":50},"reason":{"type":"string","minLength":1,"maxLength":128}},
    "retail_price_adjustment_refund":{"amount":{"type":"number","exclusiveMinimum":0,"maximum":100000},"method":{"type":"string","enum":["original","gift_card"]}},
}


def _register_order_write(name: str, extras: dict[str, Any]) -> None:
    properties={"order_id":{"type":"string","minLength":1,"maxLength":26},"expected_version":{"type":"integer","minimum":1},**extras}
    required=list(properties)
    async def handler(_trusted_context: dict | None = None, **kwargs: Any) -> dict:
        return await _execute_write(name,kwargs,_trusted_context)
    handler.__name__=name
    server.register(name=name,description=f"Retail受治理写操作：{name}",input_schema={"type":"object","properties":properties,"required":required},category="retail",effect="write",supports_idempotency=True,domain="retail",capabilities=(name.removeprefix("retail_"),),allowed_agent_types=("retail_agent",),risk_level="medium")(handler)


for _name,_extras in _ORDER_WRITE_SCHEMAS.items():
    _register_order_write(_name,_extras)


@server.register(name="retail_set_default_address",description="设置当前用户默认地址；只能确认后执行",input_schema={"type":"object","properties":{"address_id":{"type":"string","minLength":1,"maxLength":26},"expected_version":{"type":"integer","minimum":1}},"required":["address_id","expected_version"]},category="retail",effect="write",supports_idempotency=True,domain="retail",capabilities=("default_address",),allowed_agent_types=("retail_agent",),risk_level="medium")
async def retail_set_default_address(address_id: str, expected_version: int, _trusted_context: dict | None = None) -> dict:
    return await _execute_write("retail_set_default_address",locals_without_context(locals()),_trusted_context)


def locals_without_context(values: dict[str, Any]) -> dict[str, Any]:
    return {key:value for key,value in values.items() if key != "_trusted_context"}
