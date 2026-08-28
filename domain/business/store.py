"""业务存储端口及用于测试、评测的确定性内存实现。"""

from __future__ import annotations

import copy
import threading
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Protocol
from uuid import uuid4

from pkg.exceptions.exception import RequestConflictError, StorageOperationError
from pkg.security import encrypt_pii


class BusinessStore(Protocol):
    """业务服务依赖的最小存储契约。

    `get_owned`/`list_owned` 的实现必须在查询层同时约束资源 ID 与用户 ID；
    `execute_action` 必须以 `idempotency_key` 去重，并原子记录业务变更与回执。
    """

    @property
    def available(self) -> bool:
        """权威存储是否可用。"""
        ...

    def user_is_active(self, user_id: str) -> bool:
        """用户是否存在且处于可服务状态。"""
        ...

    def get_owned(self, resource: str, resource_id: str, user_id: str) -> dict[str, Any] | None:
        """同时按主键和用户所有权读取资源。"""
        ...

    def list_owned(self, resource: str, user_id: str, **filters: Any) -> list[dict[str, Any]]:
        """在用户作用域内列出满足条件的资源。"""
        ...

    def list_order_candidates(self, user_id: str) -> list[dict[str, Any]]:
        """列出带商品明细的用户订单候选。"""
        ...

    def list_public(self, resource: str, **filters: Any) -> list[dict[str, Any]]:
        """读取无需用户所有权的公共目录数据。"""
        ...

    def execute_action(self, tool_name: str, arguments: dict[str, Any], user_id: str, action_id: str, idempotency_key: str) -> dict[str, Any]:
        """幂等执行已确认业务动作并返回回执。"""
        ...

    def create_pending(self, action: dict[str, Any]) -> dict[str, Any]:
        """创建唯一活跃的待确认动作。"""
        ...

    def get_active_pending(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        """读取指定用户会话的活跃动作。"""
        ...

    def transition_pending(self, action_id: str, user_id: str, session_id: str, from_status: str, to_status: str, **updates: Any) -> dict[str, Any]:
        """使用比较并交换语义迁移动作状态。"""
        ...


class InMemoryBusinessStore:
    """测试和评测专用的确定性存储，生产启动流程不会隐式选择它。

    返回值始终深拷贝以模拟持久化边界；写路径用重入锁串行化，并保留幂等
    回执。各 `_action_*` 方法通过工具名动态分派，是注册业务动作的执行器，
    不是未调用代码。
    """

    def __init__(self, fixtures: dict[str, list[dict[str, Any]]] | None = None):
        self._data = {name: [copy.deepcopy(row) for row in rows] for name, rows in (fixtures or {}).items()}
        self._pending: dict[str, dict[str, Any]] = {}
        self._receipts: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    @property
    def available(self) -> bool:
        """内存 fixture 始终可用。"""

        return True

    def user_is_active(self, user_id: str) -> bool:
        """判断 fixture 中的用户是否存在且处于 active 状态。"""

        return any(row.get("user_id") == user_id and row.get("status") == "active" for row in self._data.get("users", []))

    def get_owned(self, resource: str, resource_id: str, user_id: str) -> dict[str, Any] | None:
        """按资源主键和递归所有权关系读取单条私有资源。"""

        id_field = _id_field(resource)
        for row in self._data.get(resource, []):
            if row.get(id_field) == resource_id and _owned(row, user_id, self._data):
                return copy.deepcopy(row)
        return None

    def list_owned(self, resource: str, user_id: str, **filters: Any) -> list[dict[str, Any]]:
        """列出用户拥有且满足精确过滤条件的资源副本。"""

        return [copy.deepcopy(row) for row in self._data.get(resource, []) if _owned(row, user_id, self._data) and _matches(row, filters)]

    def list_order_candidates(self, user_id: str) -> list[dict[str, Any]]:
        """组装订单及商品快照，并按下单时间倒序返回。"""

        orders = self.list_owned("orders", user_id)
        for order in orders:
            order["items"] = [
                copy.deepcopy(item)
                for item in self._data.get("order_items", [])
                if item.get("order_id") == order.get("order_id")
            ]
        return sorted(orders, key=lambda item: str(item.get("placed_at", "")), reverse=True)

    def list_public(self, resource: str, **filters: Any) -> list[dict[str, Any]]:
        """返回满足过滤条件的公共 fixture 数据副本。"""

        return [copy.deepcopy(row) for row in self._data.get(resource, []) if _matches(row, filters)]

    def execute_action(self, tool_name: str, arguments: dict[str, Any], user_id: str, action_id: str, idempotency_key: str) -> dict[str, Any]:
        """按工具名分派写动作，同一幂等键直接返回首次执行回执。"""

        with self._lock:
            existing = self._receipts.get(idempotency_key)
            if existing is not None:
                return copy.deepcopy(existing)
            handler = getattr(self, f"_action_{tool_name}", None)
            if handler is None:
                raise StorageOperationError()
            result = handler(copy.deepcopy(arguments), user_id, action_id)
            receipt = {"status": "succeeded", **result}
            self._receipts[idempotency_key] = receipt
            return copy.deepcopy(receipt)

    def create_pending(self, action: dict[str, Any]) -> dict[str, Any]:
        """创建待确认动作，并拒绝同一用户会话中的第二个活跃动作。"""

        with self._lock:
            if self.get_active_pending(action["user_id"], action["session_id"]):
                raise RequestConflictError()
            self._pending[action["action_id"]] = copy.deepcopy(action)
            return copy.deepcopy(action)

    def get_active_pending(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        """读取会话活跃动作，同时惰性地将超时动作标记为 expired。"""

        now = datetime.now(timezone.utc)
        for item in self._pending.values():
            expires = item["expires_at"]
            if item["user_id"] == user_id and item["session_id"] == session_id and item["status"] in {"awaiting_confirmation", "executing", "indeterminate"}:
                if expires > now:
                    return copy.deepcopy(item)
                item["status"] = "expired"
        return None

    def transition_pending(self, action_id: str, user_id: str, session_id: str, from_status: str, to_status: str, **updates: Any) -> dict[str, Any]:
        """以期望旧状态完成原子状态转换，防止重复确认和竞态覆盖。"""

        with self._lock:
            item = self._pending.get(action_id)
            if not item or item["user_id"] != user_id or item["session_id"] != session_id or item["status"] != from_status:
                raise RequestConflictError()
            if "receipt_json" in updates:
                updates["receipt"] = updates.pop("receipt_json")
            item.update(updates, status=to_status, updated_at=datetime.now(timezone.utc))
            return copy.deepcopy(item)

    def list_indeterminate(self,limit: int=20) -> list[dict[str,Any]]:
        """列出写入结果未知、等待对账的动作。"""

        return [copy.deepcopy(item) for item in self._pending.values() if item.get("status")=="indeterminate"][:limit]

    def get_receipt(self,idempotency_key: str) -> dict[str,Any] | None:
        """按幂等键读取已完成写操作的不可变回执。"""

        item=self._receipts.get(idempotency_key)
        return copy.deepcopy(item) if item else None

    def evaluation_snapshot(self) -> dict[str, Any]:
        """Return an isolated, semantic state projection for deterministic evals.

        Runtime code never uses this method for authorization.  Evaluation code uses
        it to verify business state independently from the LangGraph completion state.
        """
        with self._lock:
            snapshot = {
                table: {
                    str(row.get(_id_field(table), index)): copy.deepcopy(row)
                    for index, row in enumerate(rows)
                }
                for table, rows in self._data.items()
            }
            snapshot["governed_actions"] = copy.deepcopy(self._pending)
            snapshot["receipts"] = copy.deepcopy(self._receipts)
            return snapshot

    def _resource_for_action(self, args: dict[str, Any], user_id: str) -> tuple[str, dict[str, Any]]:
        """从动作参数解析受影响资源，并强制检查其用户所有权。"""

        for resource, key in (("lines", "line_id"), ("orders", "order_id"), ("addresses", "address_id")):
            if key in args:
                row = self.get_owned(resource, str(args[key]), user_id)
                if row is None:
                    raise StorageOperationError()
                return resource, row
        raise StorageOperationError()

    def _replace(self, resource: str, updated: dict[str, Any]) -> None:
        """按主键替换内存表中的资源记录。"""

        key = _id_field(resource)
        for index, row in enumerate(self._data.get(resource, [])):
            if row.get(key) == updated.get(key):
                self._data[resource][index] = updated
                return

    def _action_telecom_change_plan(self, args: dict[str, Any], user_id: str, action_id: str) -> dict[str, Any]:
        """校验线路版本和目标套餐后切换当前套餐。"""

        _, line = self._resource_for_action(args, user_id)
        plan = next((p for p in self._data.get("plans", []) if p.get("plan_id") == args.get("plan_id") and p.get("status") == "active"), None)
        if plan is None or line.get("version") != args.get("expected_version"):
            raise RequestConflictError()
        before = int(line["version"])
        line.update(current_plan_id=plan["plan_id"], version=before + 1, last_plan_change_at=datetime.now(timezone.utc).isoformat())
        self._replace("lines", line)
        return {"resource_type": "line", "resource_id": line["line_id"], "version_before": before, "version_after": before + 1, "summary": {"plan_id": plan["plan_id"], "plan_name": plan["name"]}}

    def _action_telecom_refuel_data(self, args: dict[str, Any], user_id: str, action_id: str) -> dict[str, Any]:
        """扣减可用余额、增加流量并记录 addon。"""

        _, line = self._resource_for_action(args, user_id)
        if line.get("version") != args.get("expected_version"):
            raise RequestConflictError()
        usage = next((u for u in self._data.get("usage_cycles", []) if u.get("line_id") == line["line_id"] and u.get("current")), None)
        if usage is None:
            raise StorageOperationError()
        wallet=next((item for item in self._data.get("wallets",[]) if item.get("account_id")==line.get("account_id")),None)
        if wallet is not None:
            charge=Decimal(str(args["quoted_price"])); balance=Decimal(str(wallet.get("available_balance",0)))
            if balance<charge: raise RequestConflictError()
            wallet["available_balance"]=str(balance-charge); wallet["version"]=int(wallet.get("version",1))+1
        usage["refueled_data_mb"] = int(usage.get("refueled_data_mb", 0)) + int(args["amount_mb"])
        self._data.setdefault("data_addons", []).append({"addon_id": uuid4().hex[:26], "line_id": line["line_id"], "amount_mb": args["amount_mb"], "charge_amount": args["quoted_price"], "action_id": action_id, "status": "applied"})
        before = int(line["version"]); line["version"] = before + 1; self._replace("lines", line)
        return {"resource_type": "line", "resource_id": line["line_id"], "version_before": before, "version_after": before + 1, "summary": {"amount_mb": args["amount_mb"], "charged": args["quoted_price"]}}

    def _action_telecom_set_roaming(self, args: dict[str, Any], user_id: str, action_id: str) -> dict[str, Any]:
        """在版本匹配时更新线路漫游状态。"""

        _, line = self._resource_for_action(args, user_id)
        if line.get("version") != args.get("expected_version"):
            raise RequestConflictError()
        before = int(line["version"]); line.update(roaming_enabled=bool(args["enabled"]), version=before + 1); self._replace("lines", line)
        return {"resource_type": "line", "resource_id": line["line_id"], "version_before": before, "version_after": before + 1, "summary": {"roaming_enabled": bool(args["enabled"])}}

    def _action_retail_cancel_order(self, args: dict[str, Any], user_id: str, action_id: str) -> dict[str, Any]:
        """仅允许取消 pending 订单。"""

        return self._update_order(args, user_id, allowed_statuses={"pending"}, status="cancelled", cancel_reason=args["reason"])

    def _action_retail_update_order_address(self, args: dict[str, Any], user_id: str, action_id: str) -> dict[str, Any]:
        """校验地址所有权后更新 pending 订单收货地址。"""

        if self.get_owned("addresses",args["address_id"],user_id) is None: raise RequestConflictError()
        return self._update_order(args, user_id, allowed_statuses={"pending"}, shipping_address_id=args["address_id"])

    def _action_retail_update_order_payment(self, args: dict[str, Any], user_id: str, action_id: str) -> dict[str, Any]:
        """校验支付方式所有权后更新 pending 订单。"""

        if self.get_owned("payment_methods",args["payment_method_id"],user_id) is None: raise RequestConflictError()
        return self._update_order(args, user_id, allowed_statuses={"pending"}, payment_method_id=args["payment_method_id"])

    def _action_retail_update_order_items(self, args: dict[str, Any], user_id: str, action_id: str) -> dict[str, Any]:
        """校验商品变体和库存后替换 pending 订单商品。"""

        for requested in args["items"]:
            variant=next((item for item in self._data.get("variants",[]) if item.get("variant_id")==requested.get("variant_id") and item.get("status")=="active"),None)
            inventory=next((item for item in self._data.get("inventory",[]) if item.get("variant_id")==requested.get("variant_id")),None)
            if variant is None or inventory is None or int(inventory.get("available_qty",0))<int(requested.get("quantity",0)): raise RequestConflictError()
        return self._update_order(args, user_id, allowed_statuses={"pending"}, items=copy.deepcopy(args["items"]), items_modified=True)

    def _action_retail_set_default_address(self, args: dict[str, Any], user_id: str, action_id: str) -> dict[str, Any]:
        """原子取消旧默认地址并设置新默认地址。"""

        address = self.get_owned("addresses", args["address_id"], user_id)
        if address is None or address.get("version") != args.get("expected_version"):
            raise RequestConflictError()
        for row in self._data.get("addresses", []):
            if row.get("user_id") == user_id:
                row["is_default"] = row.get("address_id") == address["address_id"]
        before = int(address["version"]); address["version"] = before + 1; address["is_default"] = True; self._replace("addresses", address)
        return {"resource_type": "address", "resource_id": address["address_id"], "version_before": before, "version_after": before + 1, "summary": {"is_default": True}}

    def _action_retail_create_address(self, args: dict[str, Any], user_id: str, action_id: str) -> dict[str, Any]:
        """加密敏感字段后创建地址，并可同步切换默认项。"""

        address_id = uuid4().hex[:26]
        set_default = bool(args["set_default"])
        if set_default:
            for row in self._data.get("addresses", []):
                if row.get("user_id") == user_id and row.get("status") == "active" and row.get("is_default"):
                    row["is_default"] = False
                    row["version"] = int(row.get("version", 1)) + 1
        address = {
            "address_id": address_id,
            "user_id": user_id,
            "label": args.get("label") or "默认收货地址",
            "recipient_cipher": encrypt_pii(args["recipient"]),
            "phone_cipher": encrypt_pii(args["phone"]),
            "province": args["province"],
            "city": args["city"],
            "district": args["district"],
            "detail_cipher": encrypt_pii(args["detail"]),
            "postal_code": args.get("postal_code"),
            "is_default": set_default,
            "status": "active",
            "version": 1,
        }
        self._data.setdefault("addresses", []).append(address)
        return {
            "resource_type": "address",
            "resource_id": address_id,
            "version_before": 0,
            "version_after": 1,
            "summary": {
                "label": address["label"],
                "recipient": args["recipient"],
                "phone": args["phone"],
                "full_address": (
                    f"{args['province']}{args['city']}{args['district']}{args['detail']}"
                ),
                "is_default": set_default,
            },
        }

    def _action_retail_request_return(self, args: dict[str, Any], user_id: str, action_id: str) -> dict[str, Any]:
        """验证可退数量后把已送达订单转为退货申请状态。"""

        self._validate_return_items(args)
        result = self._update_order(args, user_id, allowed_statuses={"delivered"}, status="return_requested")
        result["summary"].update({"return_id": uuid4().hex[:26], "items": args["items"], "reason": args["reason"]})
        return result

    def _action_retail_request_exchange(self, args: dict[str, Any], user_id: str, action_id: str) -> dict[str, Any]:
        """验证可换数量后把已送达订单转为换货申请状态。"""

        self._validate_return_items(args)
        result = self._update_order(args, user_id, allowed_statuses={"delivered"}, status="exchange_requested")
        result["summary"].update({"exchange_id": uuid4().hex[:26], "items": args["items"], "reason": args["reason"]})
        return result

    def _action_retail_price_adjustment_refund(self, args: dict[str, Any], user_id: str, action_id: str) -> dict[str, Any]:
        """为已送达订单创建价保退款语义回执。"""

        result = self._update_order(args, user_id, allowed_statuses={"delivered"})
        result["summary"].update({"refund_id": uuid4().hex[:26], "amount": str(Decimal(str(args["amount"]))), "method": args["method"]})
        return result

    def _update_order(self, args: dict[str, Any], user_id: str, *, allowed_statuses: set[str], **changes: Any) -> dict[str, Any]:
        """校验订单所有权、版本和前置状态后更新并生成标准回执。"""

        order = self.get_owned("orders", args["order_id"], user_id)
        if order is None or order.get("version") != args.get("expected_version"):
            raise RequestConflictError()
        if order.get("status") not in allowed_statuses:
            raise RequestConflictError()
        before = int(order["version"]); order.update(changes, version=before + 1); self._replace("orders", order)
        return {"resource_type": "order", "resource_id": order["order_id"], "version_before": before, "version_after": before + 1, "summary": changes}

    def _validate_return_items(self,args: dict[str,Any]) -> None:
        """确保本次退换数量不超过订单商品尚可售后的数量。"""

        for requested in args["items"]:
            item=next((row for row in self._data.get("order_items",[]) if row.get("order_item_id")==requested.get("order_item_id") and row.get("order_id")==args.get("order_id")),None)
            used=int(item.get("returned_qty",0))+int(item.get("exchanged_qty",0)) if item else 0
            if item is None or int(requested.get("quantity",0))+used>int(item.get("quantity",0)): raise RequestConflictError()


def _id_field(resource: str) -> str:
    """返回 fixture 表的主键字段名。"""

    return {"users": "user_id", "plans": "plan_id", "lines": "line_id", "usage_cycles": "usage_id", "products": "product_id", "variants": "variant_id", "orders": "order_id", "addresses": "address_id", "payment_methods": "payment_method_id"}.get(resource, f"{resource.rstrip('s')}_id")


def _matches(row: dict[str, Any], filters: dict[str, Any]) -> bool:
    """实现内存适配器的精确过滤语义，值为 None 时忽略该条件。"""

    return all(value is None or row.get(key) == value for key, value in filters.items())


def _owned(row: dict[str, Any], user_id: str, data: dict[str, list[dict[str, Any]]]) -> bool:
    """沿用户、账户、线路或订单关系递归判断资源所有权。"""

    if row.get("user_id") is not None:
        return row.get("user_id") == user_id
    if row.get("account_id") is not None:
        return any(account.get("account_id") == row.get("account_id") and account.get("user_id") == user_id for account in data.get("accounts", []))
    if row.get("line_id") is not None:
        line = next((item for item in data.get("lines", []) if item.get("line_id") == row.get("line_id")), None)
        return bool(line and _owned(line, user_id, data))
    if row.get("order_id") is not None:
        return any(order.get("order_id") == row.get("order_id") and order.get("user_id") == user_id for order in data.get("orders", []))
    return False
