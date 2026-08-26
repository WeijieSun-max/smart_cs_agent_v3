from __future__ import annotations

import re

from domain.customer_service_agent.orchestration.capability_index import get_capability_index
from domain.customer_service_agent.orchestration.models import RouteDecision


# 写能力词表从工具注册表派生（单一事实来源）。
WRITE_CAPABILITIES = get_capability_index().write_capabilities

# 否定词：命中写意图时视为“拒绝/decline”，不得据此生成写提案。
_NEGATION = re.compile(r"不用|不要|不想|不打算|无需|别")

# 归属/购买信号：用于区分“本人订单历史”(order_query) 与“在售商品目录”(product_query)。
_OWNERSHIP = re.compile(r"我买|买过|购买|我的订单")
_ITEM = re.compile(r"订单|商品|东西|手机|产品|平板|耳机|路由器|手表|手环")


def route_request(query: str) -> RouteDecision:
    telecom = bool(re.search(
        r"套餐|流量|通话|电话|漫游|信号|网速|SIM|APN|短信|彩信|VPN|飞行模式", query, re.I,
    ))
    colloquial_return = bool(re.search(
        r"(?:退掉|推掉|退回).*(?:订单|商品|手机|买|购买)|(?:订单|商品|手机).*(?:退掉|推掉|退回)",
        query,
    ))
    return_action = colloquial_return or bool(re.search(
        r"(?:帮我|我要|申请|办理|发起|现在).*(?:退货|换货)|(?:退货|换货).*(?:申请|办理|发起)",
        query,
    ))
    retail_policy_question = not return_action and bool(re.search(
        r"(?:退货|换货).*(?:条件|期限|政策|规则|要求|几天|多久|可以|能否)|"
        r"(?:可以|能否|是否|还能).*(?:退货|换货)|超过\s*\d+\s*天.*(?:退货|换货)|"
        r"退换货原则|配送政策|支付政策|购物说明",
        query,
    ))
    retail = bool(re.search(r"商品|商城|订单|退货|换货|退款|收货地址|支付方式|购物", query)) or colloquial_return
    # “本人订单历史”需同时命中归属信号与物品信号，避免把“有哪些在售商品”误判为订单。
    order_history = bool(_OWNERSHIP.search(query)) and bool(_ITEM.search(query))

    capabilities: list[str] = []

    # ---- telecom 能力链（互斥）----
    if re.search(r"推荐.*套餐|套餐.*推荐|套餐.*合适|流量.*不够|通话.*多.*套餐|降低月租", query):
        capabilities.append("plan_recommendation")
    elif re.search(
        r"没信号|没有信号|无信号|信号(?:差|弱|不好)|网速|上不了网|SIM|APN|短信|彩信|VPN|飞行模式",
        query, re.I,
    ):
        capabilities.append("telecom_troubleshooting")
    elif re.search(r"(?:换|改成|换成|换到|变更).*(?:套餐|plan_id)|换.*套餐|变更.*套餐", query):
        capabilities.append("plan_change")
    elif re.search(r"补.*流量|加.*流量|购买.*流量", query):
        capabilities.append("data_refuel")
    elif "漫游" in query and re.search(r"开|关|办理|启用|停用", query):
        capabilities.append("roaming")
    elif telecom and re.search(r"当前|查询|用了|使用", query):
        capabilities.append("usage" if "流量" in query or "通话" in query else "current_plan")

    # ---- retail 能力链（与 telecom 链并列，可形成跨域 composite）----
    if re.search(r"设为.*默认.*地址|默认.*收货地址", query):
        capabilities.append("default_address")
    elif re.search(r"取消.*订单", query):
        capabilities.append("cancel_order")
    elif re.search(r"修改.*地址|更换.*地址", query):
        capabilities.append("update_order_address")
    elif re.search(r"修改.*支付|更换.*支付", query):
        capabilities.append("update_order_payment")
    elif re.search(r"修改.*商品|更换.*商品|订单.*加|订单.*减", query):
        capabilities.append("update_order_items")
    elif retail_policy_question:
        capabilities.append("retail_policy")
    elif "退货" in query or colloquial_return:
        capabilities.append("request_return")
    elif "换货" in query:
        capabilities.append("request_exchange")
    elif re.search(r"差价.*退|保价|价格保护", query):
        capabilities.append("price_adjustment_refund")
    elif order_history or (retail and "订单" in query):
        capabilities.append("order_query")
    elif retail:
        capabilities.append("product_query")

    capabilities = list(dict.fromkeys(capabilities)) or ["fallback"]

    # 否定写意图：把“不换套餐/不退/不取消”等视为拒绝，路由到 fallback，绝不生成写提案。
    if _NEGATION.search(query) and any(item in WRITE_CAPABILITIES for item in capabilities):
        capabilities = ["fallback"]

    domains = tuple(_capability_domain(item) for item in capabilities)
    return RouteDecision(
        domains=domains,
        capabilities=tuple(capabilities),
        # 置信度由“是否解析到具体能力”决定：解析不到（fallback）即为低置信，触发 LLM 升级。
        confidence=0.3 if capabilities == ["fallback"] else 0.95,
        composite=len(domains) > 1 or len(capabilities) > 1,
        risk_level="medium" if any(item in WRITE_CAPABILITIES for item in capabilities) else "low",
    )


def _capability_domain(capability: str) -> str:
    if capability == "fallback":
        return "fallback"
    if capability.startswith(("plan_", "data_", "roaming", "current_", "usage", "telecom_")):
        return "telecom"
    return "retail"
