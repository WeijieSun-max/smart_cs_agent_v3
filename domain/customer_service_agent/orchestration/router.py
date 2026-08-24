from __future__ import annotations

import re

from domain.customer_service_agent.orchestration.models import RouteDecision


WRITE_CAPABILITIES = frozenset({
    "plan_change",
    "data_refuel",
    "roaming",
    "cancel_order",
    "update_order_address",
    "update_order_payment",
    "update_order_items",
    "request_return",
    "request_exchange",
    "price_adjustment_refund",
})


def route_request(query: str) -> RouteDecision:
    telecom = bool(re.search(r"套餐|流量|通话|电话|漫游|信号|网速|SIM|APN|短信|彩信|VPN|飞行模式", query, re.I))
    retail = bool(re.search(r"商品|商城|订单|退货|换货|退款|收货地址|支付方式|购物", query))
    capabilities: list[str] = []
    if re.search(r"推荐.*套餐|套餐.*推荐|套餐.*合适|流量.*不够|通话.*多.*套餐|降低月租", query):
        capabilities.append("plan_recommendation")
    elif re.search(r"没信号|无信号|网速|上不了网|SIM|APN|短信|彩信|VPN|飞行模式", query, re.I):
        capabilities.append("telecom_troubleshooting")
    elif re.search(r"换.*套餐|变更.*套餐", query):
        capabilities.append("plan_change")
    elif re.search(r"补.*流量|加.*流量|购买.*流量", query):
        capabilities.append("data_refuel")
    elif "漫游" in query and re.search(r"开|关|办理|启用|停用", query):
        capabilities.append("roaming")
    elif telecom and re.search(r"当前|查询|用了|使用", query):
        capabilities.append("usage" if "流量" in query or "通话" in query else "current_plan")
    if re.search(r"取消.*订单", query):
        capabilities.append("cancel_order")
    elif re.search(r"修改.*地址|更换.*地址", query):
        capabilities.append("update_order_address")
    elif re.search(r"修改.*支付|更换.*支付", query):
        capabilities.append("update_order_payment")
    elif re.search(r"修改.*商品|更换.*商品|订单.*加|订单.*减", query):
        capabilities.append("update_order_items")
    elif "退货" in query:
        capabilities.append("request_return")
    elif "换货" in query:
        capabilities.append("request_exchange")
    elif re.search(r"差价.*退|保价|价格保护", query):
        capabilities.append("price_adjustment_refund")
    elif retail and "订单" in query:
        capabilities.append("order_query")
    elif retail and re.search(r"退换货原则|配送|支付|购物说明", query):
        capabilities.append("retail_policy")
    elif retail:
        capabilities.append("product_query")
    domains = tuple(item for item, flag in (("telecom", telecom), ("retail", retail)) if flag) or ("fallback",)
    if not capabilities:
        capabilities = ["fallback"]
    return RouteDecision(
        domains=domains,
        capabilities=tuple(dict.fromkeys(capabilities)),
        confidence=0.95 if telecom or retail else 0.3,
        composite=len(domains) > 1 or len(capabilities) > 1,
        risk_level="medium" if any(item in WRITE_CAPABILITIES for item in capabilities) else "low",
    )
