"""面向客服工具的业务查询、所有权校验与确定性计算。"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from pkg.exceptions.exception import ServiceError, StorageUnavailableError
from pkg.security import decrypt_pii

from .models import PlanComparison, UsageProfile
from .store import BusinessStore


class UserNotActiveError(ServiceError):
    """请求用户不存在或已停用，禁止继续读取或修改其业务数据。"""

    code = "identity.user_not_active"
    status_code = 403
    safe_message = "当前用户不存在或不可用，不提供该业务服务。"


class ResourceNotFoundError(ServiceError):
    """资源不存在或不属于当前用户；统一响应以避免泄露资源是否存在。"""

    code = "business.resource_not_found"
    status_code = 404
    safe_message = "未找到属于当前用户的业务资源。"


class BusinessService:
    """在存储端口之上实施用户所有权和电信/零售业务规则。

    所有私有资源查询都先验证有效用户，并把 `user_id` 传入存储查询。服务
    仅返回属于调用者的数据；写操作仍须经过 `GovernedActionService` 的确认
    流程，本类不向 Agent 暴露绕过治理的直接入口。
    """

    def __init__(self, store: BusinessStore):
        self.store = store

    def require_available(self) -> None:
        """要求权威业务存储可用；不可用时拒绝用缓存数据冒充事实。"""

        if not self.store.available:
            raise StorageUnavailableError()

    def require_active_user(self, user_id: str) -> None:
        """验证用户存在且启用，作为所有私有业务能力的前置条件。"""

        self.require_available()
        if not self.store.user_is_active(user_id):
            raise UserNotActiveError()

    def resolve_line(self, user_id: str, line_id: str | None = None) -> dict[str, Any]:
        """解析用户线路；省略 ID 时只允许恰好存在一条活跃线路。"""

        self.require_active_user(user_id)
        if line_id:
            line = self.store.get_owned("lines", line_id, user_id)
            if line is None:
                raise ResourceNotFoundError()
            return line
        lines = self.store.list_owned("lines", user_id, status="active")
        if len(lines) != 1:
            raise ResourceNotFoundError()
        return lines[0]

    def current_plan(self, user_id: str, line_id: str | None = None) -> dict[str, Any]:
        """返回线路当前套餐及线路版本，版本供后续写操作做乐观锁。"""

        line = self.resolve_line(user_id, line_id)
        plans = self.store.list_public("plans", status="active")
        plan = next((item for item in plans if item.get("plan_id") == line.get("current_plan_id")), None)
        if plan is None:
            raise ResourceNotFoundError()
        return {"line_id": line["line_id"], "line_version": line["version"], **plan}

    def usage_profile(self, user_id: str, line_id: str | None = None) -> UsageProfile:
        """用最近三个完整账期和当前账期构造可解释的用量画像。"""

        line = self.resolve_line(user_id, line_id)
        cycles = sorted(
            self.store.list_owned("usage_cycles", user_id, line_id=line["line_id"]),
            key=lambda item: str(item.get("cycle_start", "")),
            reverse=True,
        )
        completed = [item for item in cycles if not item.get("current")][:3]
        current = next((item for item in cycles if item.get("current")), None)
        data_values = [int(item.get("used_data_mb", 0)) for item in completed]
        voice_values = [int(item.get("used_voice_minutes", 0)) for item in completed]
        projected_data = _project(current, "used_data_mb")
        projected_voice = _project(current, "used_voice_minutes")
        usable = len(completed)
        quality = "insufficient" if usable == 0 else "low_confidence" if usable == 1 else "normal_confidence"
        peak_data = max([*data_values, projected_data], default=0)
        peak_voice = max([*voice_values, projected_voice], default=0)
        current_included_data = int((current or {}).get("included_data_mb", 0))
        current_used_data = int((current or {}).get("used_data_mb", 0))
        current_refueled_data = int((current or {}).get("refueled_data_mb", 0))
        current_included_voice = int((current or {}).get("included_voice_minutes", 0))
        current_used_voice = int((current or {}).get("used_voice_minutes", 0))
        return UsageProfile(
            line_id=line["line_id"],
            current_cycle_start=_optional_text((current or {}).get("cycle_start")),
            current_cycle_end=_optional_text((current or {}).get("cycle_end")),
            current_included_data_mb=current_included_data,
            current_used_data_mb=current_used_data,
            current_refueled_data_mb=current_refueled_data,
            current_remaining_data_mb=max(
                0,
                current_included_data + current_refueled_data - current_used_data,
            ),
            current_included_voice_minutes=current_included_voice,
            current_used_voice_minutes=current_used_voice,
            current_remaining_voice_minutes=max(0, current_included_voice - current_used_voice),
            usable_cycles=usable,
            analysis_period=f"最近{usable}个完整账期" + ("及当前账期预测" if current else ""),
            data_quality=quality,
            average_data_mb=_average(data_values),
            peak_data_mb=peak_data,
            projected_data_mb=projected_data,
            recommended_data_mb=math.ceil(peak_data * 1.15),
            average_voice_minutes=_average(voice_values),
            peak_voice_minutes=peak_voice,
            projected_voice_minutes=projected_voice,
            recommended_voice_minutes=math.ceil(peak_voice * 1.10),
            refuel_count=sum(int(item.get("refuel_count", 0)) for item in cycles[:3]),
        )

    def list_plans(self, user_id: str, line_id: str | None = None) -> list[dict[str, Any]]:
        """验证线路所有权后列出当前有效的公共套餐。"""

        self.resolve_line(user_id, line_id)
        return self.store.list_public("plans", status="active")

    def compare_plans(self, user_id: str, line_id: str | None, candidate_plan_ids: list[str]) -> list[dict[str, Any]]:
        """按预测月成本、容量余量和支配关系比较候选套餐。"""

        profile = self.usage_profile(user_id, line_id)
        plans = [item for item in self.list_plans(user_id, line_id) if item.get("plan_id") in set(candidate_plan_ids)]
        comparisons: list[PlanComparison] = []
        for plan in plans:
            data_limit = int(plan.get("data_limit_mb", 0))
            voice_limit = int(plan.get("included_voice_minutes", 0))
            data_short = 0 if plan.get("data_unlimited") else max(0, profile.recommended_data_mb - data_limit)
            voice_short = 0 if plan.get("voice_unlimited") else max(0, profile.recommended_voice_minutes - voice_limit)
            data_cost = (Decimal(math.ceil(data_short / 1024)) * Decimal(str(plan.get("refuel_price_per_gb", 0)))) if data_short else Decimal("0")
            voice_cost = Decimal(voice_short) * Decimal(str(plan.get("voice_overage_price_per_minute", 0)))
            price = Decimal(str(plan["monthly_price"]))
            comparisons.append(PlanComparison(
                plan_id=str(plan["plan_id"]), name=str(plan["name"]), eligible=True,
                monthly_price=price,
                expected_monthly_cost=(price + data_cost + voice_cost).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                meets_data_need=data_short == 0,
                meets_voice_need=voice_short == 0,
                data_buffer_mb=10**12 if plan.get("data_unlimited") else data_limit - profile.recommended_data_mb,
                voice_buffer_minutes=10**9 if plan.get("voice_unlimited") else voice_limit - profile.recommended_voice_minutes,
            ))
        for left in comparisons:
            left.dominated = any(
                right.plan_id != left.plan_id
                and right.expected_monthly_cost <= left.expected_monthly_cost
                and right.data_buffer_mb >= left.data_buffer_mb
                and right.voice_buffer_minutes >= left.voice_buffer_minutes
                and (right.expected_monthly_cost < left.expected_monthly_cost or right.data_buffer_mb > left.data_buffer_mb or right.voice_buffer_minutes > left.voice_buffer_minutes)
                for right in comparisons
            )
        return [item.model_dump(mode="json") for item in sorted(comparisons, key=lambda item: (item.dominated, item.expected_monthly_cost))]

    def get_order(self, user_id: str, order_id: str) -> dict[str, Any]:
        """按所有权读取订单主记录。"""

        self.require_active_user(user_id)
        order = self.store.get_owned("orders", order_id, user_id)
        if order is None:
            raise ResourceNotFoundError()
        return order

    def get_order_detail(self, user_id: str, order_id: str) -> dict[str, Any]:
        """读取订单及商品快照明细，用于售后消歧与数量校验。"""

        self.require_active_user(user_id)
        order = next(
            (item for item in self.store.list_order_candidates(user_id) if item.get("order_id") == order_id),
            None,
        )
        if order is None:
            raise ResourceNotFoundError()
        return order

    def list_orders(self, user_id: str, status: str | None = None) -> list[dict[str, Any]]:
        """列出当前用户订单，可按状态精确过滤。"""

        self.require_active_user(user_id)
        return self.store.list_owned("orders", user_id, status=status)

    def find_orders(
        self,
        user_id: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        product_query: str = "",
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """按半开日期区间、商品文本和状态筛选最多 50 个订单候选。"""

        self.require_active_user(user_id)
        normalized_product = product_query.strip().lower()
        result = []
        for order in self.store.list_order_candidates(user_id):
            placed_date = str(order.get("placed_at") or "")[:10]
            if start_date and placed_date < start_date:
                continue
            if end_date and placed_date >= end_date:
                continue
            if status and order.get("status") != status:
                continue
            if normalized_product and not any(
                normalized_product in f"{item.get('name_snapshot', '')} {item.get('sku_snapshot', '')}".lower()
                for item in order.get("items", [])
            ):
                continue
            result.append(order)
        return result[:50]

    def list_products(self, query: str = "") -> list[dict[str, Any]]:
        """查询公共在售商品；文本匹配仅用于缩小候选，不参与授权。"""

        products = self.store.list_public("products", status="active")
        if not query.strip():
            return products
        normalized = query.lower()
        return [item for item in products if normalized in f"{item.get('name', '')} {item.get('description', '')}".lower()]

    def list_addresses(self,user_id: str) -> list[dict[str,Any]]:
        """返回用户有效地址，并在领域边界内解密受保护字段。"""

        self.require_active_user(user_id)
        return [
            _address_for_customer(item)
            for item in self.store.list_owned("addresses",user_id,status="active")
        ]

    def get_default_address(self, user_id: str) -> dict[str, Any]:
        """返回当前用户唯一默认地址及完整联系人，避免模型自行筛选地址列表。"""

        addresses = self.list_addresses(user_id)
        # MySQL BOOLEAN is backed by TINYINT and PyMySQL returns 0/1 integers,
        # while the in-memory adapter uses real bool values.  Compare by value
        # so the authoritative database row is not mistaken for no default.
        defaults = [item for item in addresses if item.get("is_default") == 1]
        if not defaults:
            return {"status": "not_found", "address": None}
        if len(defaults) > 1:
            return {
                "status": "ambiguous",
                "address": None,
                "candidate_count": len(defaults),
            }
        return {"status": "found", "address": defaults[0]}

    def list_payment_methods(self,user_id: str) -> list[dict[str,Any]]:
        """返回用户有效的脱敏支付方式。"""

        self.require_active_user(user_id)
        return self.store.list_owned("payment_methods",user_id,status="active")

    def execute_action(self, tool_name: str, arguments: dict[str, Any], user_id: str, action_id: str, idempotency_key: str) -> dict[str, Any]:
        """在再次验证用户后把已治理写操作交给权威存储执行。"""

        self.require_active_user(user_id)
        return self.store.execute_action(tool_name, arguments, user_id, action_id, idempotency_key)


def _average(values: list[int]) -> int:
    """计算四舍五入后的整数平均值；空样本返回零。"""

    return round(sum(values) / len(values)) if values else 0


def _optional_text(value: Any) -> str | None:
    """把可选日期/标量转换为非空文本，供 JSON 领域模型稳定输出。"""

    text = str(value).strip() if value is not None else ""
    return text or None


def _project(current: dict[str, Any] | None, field: str) -> int:
    """按已过天数外推当前账期；不足七天时避免放大早期噪声。"""

    if not current:
        return 0
    elapsed = max(1, int(current.get("elapsed_days", 0)))
    total = max(elapsed, int(current.get("cycle_days", 30)))
    if elapsed < 7:
        return int(current.get(field, 0))
    return math.ceil(int(current.get(field, 0)) * total / elapsed)


def _address_for_customer(address: dict[str, Any]) -> dict[str, Any]:
    """Expose owned address details to the customer without leaking cipher blobs."""
    output = {
        key: value
        for key, value in address.items()
        if key not in {"recipient_cipher", "phone_cipher", "detail_cipher"}
    }
    protected_fields = {
        "recipient": "recipient_cipher",
        "phone": "phone_cipher",
        "detail": "detail_cipher",
    }
    for public_name, cipher_name in protected_fields.items():
        value = address.get(public_name)
        if not isinstance(value, str) or not value:
            value = decrypt_pii(address.get(cipher_name))
        output[public_name] = value or "[历史数据不可恢复，请重新保存]"
    return output


_service: BusinessService | None = None


def initialize_service(store: BusinessStore) -> BusinessService:
    """安装使用指定存储端口的进程级业务服务。"""

    global _service
    _service = BusinessService(store)
    return _service


def get_service() -> BusinessService:
    """取得已初始化的业务服务。"""

    if _service is None:
        raise RuntimeError("business service is not initialized")
    return _service
