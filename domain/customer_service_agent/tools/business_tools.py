"""注册电信与零售读工具，以及只能由治理动作触发的幂等写工具。"""

from __future__ import annotations

import asyncio
from typing import Any

from domain.business import service as business_service
from pkg.exceptions.exception import ToolValidationError

from .tool_registry import server


def _user(context: dict[str, Any] | None) -> str:
    """从可信工具上下文读取用户 ID，缺失或非法时关闭调用。"""

    user_id = (context or {}).get("user_id")
    if not isinstance(user_id, str) or not user_id:
        raise ToolValidationError()
    return user_id


def _write_context(context: dict[str, Any] | None) -> tuple[str, str, str]:
    """提取写执行所需的用户、治理动作和幂等标识。"""

    user_id = _user(context)
    action_id = (context or {}).get("action_id")
    idempotency_key = (context or {}).get("idempotency_key")
    if not isinstance(action_id, str) or not isinstance(idempotency_key, str):
        raise ToolValidationError()
    return user_id, action_id, idempotency_key


_LINE_SCHEMA = {
    "type": "object",
    "properties": {
        "line_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": 26,
            "description": "可选；省略时按可信 user_id 自动解析唯一活跃线路，不得要求用户重复提供",
        }
    },
}


@server.register(name="telecom_get_current_plan", description="查询当前用户线路和当前套餐", input_schema=_LINE_SCHEMA, category="telecom", effect="read", supports_idempotency=False, domain="telecom", capabilities=("current_plan", "plan_recommendation"), allowed_agent_types=("telecom_agent",), parallel_safe=True)
async def telecom_get_current_plan(line_id: str | None = None, _trusted_context: dict | None = None) -> dict:
    """读取用户线路当前套餐及乐观锁版本。"""

    return await asyncio.to_thread(business_service.get_service().current_plan, _user(_trusted_context), line_id)


@server.register(name="telecom_get_usage_profile", description="按近期账期聚合流量和语音通话使用画像", input_schema=_LINE_SCHEMA, category="telecom", effect="read", supports_idempotency=False, domain="telecom", capabilities=("usage", "plan_recommendation"), allowed_agent_types=("telecom_agent",), parallel_safe=True)
async def telecom_get_usage_profile(line_id: str | None = None, _trusted_context: dict | None = None) -> dict:
    """返回由历史账期确定性计算的线路用量画像。"""

    profile = await asyncio.to_thread(business_service.get_service().usage_profile, _user(_trusted_context), line_id)
    return profile.model_dump(mode="json")


@server.register(name="telecom_list_plans", description="列出当前线路可选的有效套餐", input_schema=_LINE_SCHEMA, category="telecom", effect="read", supports_idempotency=False, domain="telecom", capabilities=("plan_catalog", "plan_recommendation"), allowed_agent_types=("telecom_agent",), parallel_safe=True)
async def telecom_list_plans(line_id: str | None = None, _trusted_context: dict | None = None) -> list[dict]:
    """验证线路所有权后列出有效套餐。"""

    return await asyncio.to_thread(business_service.get_service().list_plans, _user(_trusted_context), line_id)


@server.register(name="telecom_compare_plans", description="确定性比较套餐容量和预计月成本", input_schema={"type":"object","properties":{"line_id":{"type":"string","minLength":1,"maxLength":26},"candidate_plan_ids":{"type":"array","items":{"type":"string","minLength":1,"maxLength":26},"minItems":1,"maxItems":20,"uniqueItems":True}},"required":["candidate_plan_ids"]}, category="telecom", effect="read", supports_idempotency=False, domain="telecom", capabilities=("plan_recommendation",), allowed_agent_types=("telecom_agent",), parallel_safe=True)
async def telecom_compare_plans(candidate_plan_ids: list[str], line_id: str | None = None, _trusted_context: dict | None = None) -> list[dict]:
    """比较指定套餐的预测成本、容量余量和支配关系。"""

    return await asyncio.to_thread(business_service.get_service().compare_plans, _user(_trusted_context), line_id, candidate_plan_ids)


@server.register(name="telecom_quote_plan_change", description="读取套餐变更前的最新线路版本和价格影响", input_schema={"type":"object","properties":{"line_id":{"type":"string","minLength":1,"maxLength":26},"plan_id":{"type":"string","minLength":1,"maxLength":26}},"required":["line_id","plan_id"]}, category="telecom", effect="read", supports_idempotency=False, domain="telecom", capabilities=("plan_change",), allowed_agent_types=("telecom_agent",), parallel_safe=False)
async def telecom_quote_plan_change(line_id: str, plan_id: str, _trusted_context: dict | None = None) -> dict:
    """在写提议前读取目标套餐影响和最新线路版本。"""

    service=business_service.get_service(); user_id=_user(_trusted_context); current=await asyncio.to_thread(service.current_plan,user_id,line_id); plans=await asyncio.to_thread(service.list_plans,user_id,line_id); target=next((p for p in plans if p.get("plan_id")==plan_id),None)
    if target is None: raise ToolValidationError()
    return {"line_id":line_id,"plan_id":plan_id,"expected_version":current["line_version"],"current_plan":{"plan_id":current["plan_id"],"name":current["name"],"monthly_price":current["monthly_price"]},"target_plan":{"name":target["name"],"monthly_price":target["monthly_price"]}}


@server.register(name="telecom_quote_refuel", description="查询补充流量的确定价格和最新线路版本", input_schema={"type":"object","properties":{"line_id":{"type":"string","minLength":1,"maxLength":26},"amount_mb":{"type":"integer","minimum":1024,"maximum":102400,"multipleOf":1024}},"required":["line_id","amount_mb"]}, category="telecom", effect="read", supports_idempotency=False, domain="telecom", capabilities=("data_refuel",), allowed_agent_types=("telecom_agent",), parallel_safe=False)
async def telecom_quote_refuel(line_id: str, amount_mb: int, _trusted_context: dict | None = None) -> dict:
    """在写提议前计算流量包价格并读取最新线路版本。"""

    current=await asyncio.to_thread(business_service.get_service().current_plan,_user(_trusted_context),line_id); price=float(current.get("refuel_price_per_gb",0))*amount_mb/1024
    return {"line_id":line_id,"amount_mb":amount_mb,"quoted_price":round(price,2),"currency":current.get("currency","CNY"),"expected_version":current["line_version"]}


async def _execute_write(tool_name: str, arguments: dict[str, Any], context: dict | None) -> dict:
    """只使用治理服务注入的写上下文调用业务执行器。"""

    user_id, action_id, key = _write_context(context)
    return await asyncio.to_thread(business_service.get_service().execute_action, tool_name, arguments, user_id, action_id, key)


@server.register(name="telecom_change_plan", description="变更线路套餐；只能由确认后的治理动作执行", input_schema={"type":"object","properties":{"line_id":{"type":"string","minLength":1,"maxLength":26},"plan_id":{"type":"string","minLength":1,"maxLength":26},"expected_version":{"type":"integer","minimum":1}},"required":["line_id","plan_id","expected_version"]}, category="telecom", effect="write", supports_idempotency=True, domain="telecom", capabilities=("plan_change",), allowed_agent_types=("telecom_agent",), risk_level="medium")
async def telecom_change_plan(line_id: str, plan_id: str, expected_version: int, _trusted_context: dict | None = None) -> dict:
    """执行已确认、版本匹配的套餐变更。"""

    return await _execute_write(
        "telecom_change_plan",
        {"line_id": line_id, "plan_id": plan_id, "expected_version": expected_version},
        _trusted_context,
    )


@server.register(name="telecom_refuel_data", description="购买并补充流量；只能由确认后的治理动作执行", input_schema={"type":"object","properties":{"line_id":{"type":"string","minLength":1,"maxLength":26},"amount_mb":{"type":"integer","minimum":1024,"maximum":102400,"multipleOf":1024},"quoted_price":{"type":"number","minimum":0,"maximum":100000},"expected_version":{"type":"integer","minimum":1}},"required":["line_id","amount_mb","quoted_price","expected_version"]}, category="telecom", effect="write", supports_idempotency=True, domain="telecom", capabilities=("data_refuel",), allowed_agent_types=("telecom_agent",), risk_level="medium")
async def telecom_refuel_data(line_id: str, amount_mb: int, quoted_price: float, expected_version: int, _trusted_context: dict | None = None) -> dict:
    """执行已确认的流量购买，并传递冻结报价。"""

    return await _execute_write(
        "telecom_refuel_data",
        {
            "line_id": line_id,
            "amount_mb": amount_mb,
            "quoted_price": quoted_price,
            "expected_version": expected_version,
        },
        _trusted_context,
    )


@server.register(name="telecom_set_roaming", description="开启或关闭线路漫游；只能由确认后的治理动作执行", input_schema={"type":"object","properties":{"line_id":{"type":"string","minLength":1,"maxLength":26},"enabled":{"type":"boolean"},"expected_version":{"type":"integer","minimum":1}},"required":["line_id","enabled","expected_version"]}, category="telecom", effect="write", supports_idempotency=True, domain="telecom", capabilities=("roaming",), allowed_agent_types=("telecom_agent",), risk_level="medium")
async def telecom_set_roaming(line_id: str, enabled: bool, expected_version: int, _trusted_context: dict | None = None) -> dict:
    """执行已确认、版本匹配的漫游状态变更。"""

    return await _execute_write(
        "telecom_set_roaming",
        {"line_id": line_id, "enabled": enabled, "expected_version": expected_version},
        _trusted_context,
    )


@server.register(name="retail_list_products", description="查询在售商品", input_schema={"type":"object","properties":{"query":{"type":"string","maxLength":255}}}, category="retail", effect="read", supports_idempotency=False, domain="retail", capabilities=("product_query",), allowed_agent_types=("retail_agent",), parallel_safe=True)
async def retail_list_products(query: str = "", _trusted_context: dict | None = None) -> list[dict]:
    """查询公共在售商品。"""

    return await asyncio.to_thread(business_service.get_service().list_products,query)


@server.register(name="retail_get_order", description="查询属于当前用户的订单", input_schema={"type":"object","properties":{"order_id":{"type":"string","minLength":1,"maxLength":26}},"required":["order_id"]}, category="retail", effect="read", supports_idempotency=False, domain="retail", capabilities=("order_query",), allowed_agent_types=("retail_agent",), parallel_safe=True)
async def retail_get_order(order_id: str, _trusted_context: dict | None = None) -> dict:
    """读取属于当前用户的订单主记录。"""

    return await asyncio.to_thread(business_service.get_service().get_order,_user(_trusted_context),order_id)


@server.register(name="retail_get_order_detail", description="查询属于当前用户的订单及商品明细", input_schema={"type":"object","properties":{"order_id":{"type":"string","minLength":1,"maxLength":26}},"required":["order_id"]}, category="retail", effect="read", supports_idempotency=False, domain="retail", capabilities=("order_resolution","order_query","request_return","request_exchange"), allowed_agent_types=("retail_agent",), parallel_safe=True)
async def retail_get_order_detail(order_id: str, _trusted_context: dict | None = None) -> dict:
    """读取用户订单及商品快照，用于售后校验。"""

    return await asyncio.to_thread(business_service.get_service().get_order_detail,_user(_trusted_context),order_id)


@server.register(name="retail_list_orders", description="列出当前用户订单", input_schema={"type":"object","properties":{"status":{"type":"string","enum":["pending","processed","delivered","cancelled","return_requested","exchange_requested"]}}}, category="retail", effect="read", supports_idempotency=False, domain="retail", capabilities=("order_query",), allowed_agent_types=("retail_agent",), parallel_safe=True)
async def retail_list_orders(status: str | None = None, _trusted_context: dict | None = None) -> list[dict]:
    """按可选状态列出当前用户订单。"""

    return await asyncio.to_thread(business_service.get_service().list_orders,_user(_trusted_context),status)


@server.register(name="retail_find_orders", description="按数据库中的购买时间、商品和状态查找当前用户订单候选", input_schema={"type":"object","properties":{"start_date":{"type":"string","format":"date"},"end_date":{"type":"string","format":"date"},"product_query":{"type":"string","maxLength":255},"status":{"type":"string","enum":["pending","processed","delivered","cancelled","return_requested","exchange_requested"]}},"additionalProperties":False}, category="retail", effect="read", supports_idempotency=False, domain="retail", capabilities=("order_resolution","order_query"), allowed_agent_types=("retail_agent",), parallel_safe=True)
async def retail_find_orders(start_date: str | None = None, end_date: str | None = None, product_query: str = "", status: str | None = None, _trusted_context: dict | None = None) -> list[dict]:
    """按日期、商品和状态解析用户订单候选。"""

    return await asyncio.to_thread(
        business_service.get_service().find_orders,
        _user(_trusted_context),
        start_date=start_date,
        end_date=end_date,
        product_query=product_query,
        status=status,
    )


@server.register(name="retail_list_addresses",description="列出当前用户可用收货地址，包括收货人、完整联系电话、完整地址和默认状态",input_schema={"type":"object","properties":{}},category="retail",effect="read",supports_idempotency=False,domain="retail",capabilities=("address_query","default_address","create_address"),allowed_agent_types=("retail_agent",),parallel_safe=True)
async def retail_list_addresses(_trusted_context: dict | None=None) -> list[dict]:
    """列出当前用户有效收货地址。"""

    return await asyncio.to_thread(business_service.get_service().list_addresses,_user(_trusted_context))


@server.register(
    name="retail_get_default_address",
    description=(
        "读取当前用户唯一默认收货地址，包括完整收件人、联系电话、地址、默认状态和版本；"
        "修改默认地址且用户未要求变更联系人时，应先调用本工具并沿用联系人"
    ),
    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    category="retail",
    effect="read",
    supports_idempotency=False,
    domain="retail",
    capabilities=("address_query", "default_address", "create_address"),
    allowed_agent_types=("retail_agent",),
    parallel_safe=True,
)
async def retail_get_default_address(_trusted_context: dict | None = None) -> dict:
    """读取可信身份绑定用户的默认地址，不接受模型提供的用户标识。"""

    return await asyncio.to_thread(
        business_service.get_service().get_default_address,
        _user(_trusted_context),
    )


@server.register(name="retail_list_payment_methods",description="列出当前用户可用支付方式的脱敏信息",input_schema={"type":"object","properties":{}},category="retail",effect="read",supports_idempotency=False,domain="retail",capabilities=("payment_query",),allowed_agent_types=("retail_agent",),parallel_safe=True)
async def retail_list_payment_methods(_trusted_context: dict | None=None) -> list[dict]:
    """列出当前用户有效的脱敏支付方式。"""

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
    """用公共订单/版本字段批量注册同构零售写处理器。"""

    properties={"order_id":{"type":"string","minLength":1,"maxLength":26},"expected_version":{"type":"integer","minimum":1},**extras}
    required=list(properties)
    async def handler(_trusted_context: dict | None = None, **kwargs: Any) -> dict:
        """把动态注册的订单写参数交给统一治理执行路径。"""

        return await _execute_write(name,kwargs,_trusted_context)
    handler.__name__=name
    server.register(name=name,description=f"Retail受治理写操作：{name}",input_schema={"type":"object","properties":properties,"required":required},category="retail",effect="write",supports_idempotency=True,domain="retail",capabilities=(name.removeprefix("retail_"),),allowed_agent_types=("retail_agent",),risk_level="medium")(handler)


for _name,_extras in _ORDER_WRITE_SCHEMAS.items():
    _register_order_write(_name,_extras)


@server.register(name="retail_set_default_address",description="设置当前用户默认地址；只能确认后执行",input_schema={"type":"object","properties":{"address_id":{"type":"string","minLength":1,"maxLength":26},"expected_version":{"type":"integer","minimum":1}},"required":["address_id","expected_version"]},category="retail",effect="write",supports_idempotency=True,domain="retail",capabilities=("default_address",),allowed_agent_types=("retail_agent",),risk_level="medium")
async def retail_set_default_address(address_id: str, expected_version: int, _trusted_context: dict | None = None) -> dict:
    """执行已确认的默认地址切换。"""

    return await _execute_write(
        "retail_set_default_address",
        {"address_id": address_id, "expected_version": expected_version},
        _trusted_context,
    )


_CREATE_ADDRESS_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "minLength": 1, "maxLength": 64},
        "recipient": {"type": "string", "minLength": 1, "maxLength": 128},
        "phone": {"type": "string", "minLength": 7, "maxLength": 32},
        "province": {"type": "string", "minLength": 1, "maxLength": 64},
        "city": {"type": "string", "minLength": 1, "maxLength": 64},
        "district": {"type": "string", "minLength": 1, "maxLength": 64},
        "detail": {"type": "string", "minLength": 1, "maxLength": 512},
        "postal_code": {"type": ["string", "null"], "maxLength": 20},
        "set_default": {"type": "boolean"},
    },
    "required": [
        "recipient", "phone", "province", "city", "district",
        "detail", "set_default",
    ],
}


@server.register(
    name="retail_create_address",
    description="新增当前用户收货地址，可在同一次确认写操作中设为默认地址",
    input_schema=_CREATE_ADDRESS_SCHEMA,
    category="retail",
    effect="write",
    supports_idempotency=True,
    domain="retail",
    capabilities=("create_address", "default_address"),
    allowed_agent_types=("retail_agent",),
    risk_level="medium",
)
async def retail_create_address(
    recipient: str,
    phone: str,
    province: str,
    city: str,
    district: str,
    detail: str,
    set_default: bool,
    label: str = "默认收货地址",
    postal_code: str | None = None,
    _trusted_context: dict | None = None,
) -> dict:
    """执行已确认的地址创建；敏感字段由业务存储加密落库。"""

    return await _execute_write(
        "retail_create_address",
        {
            "recipient": recipient,
            "phone": phone,
            "province": province,
            "city": city,
            "district": district,
            "detail": detail,
            "set_default": set_default,
            "label": label,
            "postal_code": postal_code,
        },
        _trusted_context,
    )
