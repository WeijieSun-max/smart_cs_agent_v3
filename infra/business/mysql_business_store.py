from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import uuid4

from infra.db.mysql_client import MySQLClient
from pkg.exceptions.exception import RequestConflictError, StorageOperationError, StorageUnavailableError
from pkg.security import decrypt_pii, encrypt_pii


class MySQLBusinessStore:
    """MySQL fact-source adapter. Every personalized query includes user ownership."""

    def __init__(self, client: MySQLClient | None):
        self._client = client

    @property
    def available(self) -> bool:
        return self._client is not None

    def _require(self) -> MySQLClient:
        if self._client is None:
            raise StorageUnavailableError()
        return self._client

    def user_is_active(self, user_id: str) -> bool:
        ok, row = self._require().execute_query(
            "SELECT user_id FROM cs_users WHERE user_id=%s AND status='active' AND deleted_at IS NULL LIMIT 1",
            (user_id,), fetch_one=True,
        )
        return bool(ok and row)

    def get_owned(self, resource: str, resource_id: str, user_id: str) -> dict[str, Any] | None:
        queries = {
            "lines": "SELECT l.* FROM tc_lines l JOIN tc_accounts a ON a.account_id=l.account_id WHERE l.line_id=%s AND a.user_id=%s LIMIT 1",
            "orders": "SELECT * FROM rt_orders WHERE order_id=%s AND user_id=%s LIMIT 1",
            "addresses": "SELECT * FROM cs_user_addresses WHERE address_id=%s AND user_id=%s AND status='active' LIMIT 1",
            "payment_methods": "SELECT * FROM cs_payment_methods WHERE payment_method_id=%s AND user_id=%s AND status='active' LIMIT 1",
            "usage_cycles": "SELECT u.* FROM tc_usage_cycles u JOIN tc_lines l ON l.line_id=u.line_id JOIN tc_accounts a ON a.account_id=l.account_id WHERE u.usage_id=%s AND a.user_id=%s LIMIT 1",
        }
        sql = queries.get(resource)
        if sql is None:
            raise ValueError(f"unsupported owned resource: {resource}")
        ok, row = self._require().execute_query(sql, (resource_id, user_id), fetch_one=True)
        if not ok:
            raise StorageOperationError()
        return _normalize(row) if row else None

    def list_owned(self, resource: str, user_id: str, **filters: Any) -> list[dict[str, Any]]:
        if resource == "lines":
            sql = "SELECT l.* FROM tc_lines l JOIN tc_accounts a ON a.account_id=l.account_id WHERE a.user_id=%s"
            args: list[Any] = [user_id]
        elif resource == "usage_cycles":
            sql = "SELECT u.*, (u.cycle_end >= UTC_TIMESTAMP(6)) AS current, GREATEST(1,DATEDIFF(UTC_TIMESTAMP(6),u.cycle_start)+1) AS elapsed_days, GREATEST(1,DATEDIFF(u.cycle_end,u.cycle_start)+1) AS cycle_days FROM tc_usage_cycles u JOIN tc_lines l ON l.line_id=u.line_id JOIN tc_accounts a ON a.account_id=l.account_id WHERE a.user_id=%s"
            args = [user_id]
        elif resource == "orders":
            sql = "SELECT * FROM rt_orders WHERE user_id=%s"
            args = [user_id]
        elif resource == "addresses":
            sql = "SELECT * FROM cs_user_addresses WHERE user_id=%s"
            args = [user_id]
        elif resource == "payment_methods":
            sql = "SELECT * FROM cs_payment_methods WHERE user_id=%s"
            args = [user_id]
        else:
            raise ValueError(f"unsupported owned resource: {resource}")
        allowed = {"status", "line_id"}
        for key, value in filters.items():
            if key not in allowed or value is None:
                continue
            sql += f" AND {('u.' if resource == 'usage_cycles' else 'l.' if resource == 'lines' else '')}{key}=%s"
            args.append(value)
        ok, rows = self._require().execute_query(sql, tuple(args))
        if not ok:
            raise StorageOperationError()
        return [_normalize(row) for row in rows]

    def list_order_candidates(self, user_id: str) -> list[dict[str, Any]]:
        ok, rows = self._require().execute_query(
            """SELECT o.*, i.order_item_id, i.product_id AS item_product_id,
            i.variant_id AS item_variant_id, i.sku_snapshot, i.name_snapshot,
            i.unit_price, i.quantity, i.returned_qty, i.exchanged_qty,
            i.item_status, i.version AS item_version
            FROM rt_orders o
            LEFT JOIN rt_order_items i ON i.order_id=o.order_id
            WHERE o.user_id=%s
            ORDER BY o.placed_at DESC,o.order_id,i.order_item_id""",
            (user_id,),
        )
        if not ok:
            raise StorageOperationError()
        grouped: dict[str, dict[str, Any]] = {}
        item_fields = {
            "order_item_id", "item_product_id", "item_variant_id", "sku_snapshot",
            "name_snapshot", "unit_price", "quantity", "returned_qty",
            "exchanged_qty", "item_status", "item_version",
        }
        for raw in rows:
            row = _normalize(raw)
            order_id = str(row["order_id"])
            order = grouped.get(order_id)
            if order is None:
                order = {key: value for key, value in row.items() if key not in item_fields}
                order["items"] = []
                grouped[order_id] = order
            if row.get("order_item_id"):
                order["items"].append({
                    "order_item_id": row["order_item_id"],
                    "product_id": row.get("item_product_id"),
                    "variant_id": row.get("item_variant_id"),
                    "sku_snapshot": row.get("sku_snapshot"),
                    "name_snapshot": row.get("name_snapshot"),
                    "unit_price": row.get("unit_price"),
                    "quantity": row.get("quantity"),
                    "returned_qty": row.get("returned_qty"),
                    "exchanged_qty": row.get("exchanged_qty"),
                    "item_status": row.get("item_status"),
                    "version": row.get("item_version"),
                })
        return list(grouped.values())

    def list_public(self, resource: str, **filters: Any) -> list[dict[str, Any]]:
        table = {"plans": "tc_plans", "products": "rt_products", "variants": "rt_product_variants"}.get(resource)
        if table is None:
            raise ValueError(f"unsupported public resource: {resource}")
        sql = f"SELECT * FROM {table} WHERE 1=1"
        args: list[Any] = []
        for key, value in filters.items():
            if key not in {"status", "product_id"} or value is None:
                continue
            sql += f" AND {key}=%s"; args.append(value)
        ok, rows = self._require().execute_query(sql, tuple(args))
        if not ok:
            raise StorageOperationError()
        return [_normalize(row) for row in rows]

    def create_pending(self, action: dict[str, Any]) -> dict[str, Any]:
        client = self._require()
        existing = self.get_active_pending(action["user_id"], action["session_id"])
        if existing:
            raise RequestConflictError()
        ok, affected = client.execute_update(
            """INSERT INTO cs_governed_actions
            (action_id,user_id,session_id,turn_id,tool_name,tool_version,skill_name,skill_version,
             arguments_json,arguments_digest,impact_summary,status,idempotency_key,resource_version,
             created_at,updated_at,expires_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,UTC_TIMESTAMP(6),UTC_TIMESTAMP(6),%s)""",
            (action["action_id"], action["user_id"], action["session_id"], action["turn_id"], action["tool_name"],
             action["tool_version"], action.get("skill_name"), action.get("skill_version"),
             _encode_sensitive_json(action["arguments"]), action["arguments_digest"],
             action["impact_summary"], action["status"], action["idempotency_key"], action.get("resource_version"),
             _naive(action["expires_at"])),
        )
        if not ok or int(affected) != 1:
            raise StorageOperationError()
        return dict(action)

    def get_active_pending(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        ok, row = self._require().execute_query(
            """SELECT * FROM cs_governed_actions WHERE user_id=%s AND session_id=%s
            AND status IN ('awaiting_confirmation','executing','indeterminate') AND expires_at>UTC_TIMESTAMP(6)
            ORDER BY created_at DESC LIMIT 1""", (user_id, session_id), fetch_one=True,
        )
        if not ok:
            raise StorageOperationError()
        return _action_row(row) if row else None

    def transition_pending(self, action_id: str, user_id: str, session_id: str, from_status: str, to_status: str, **updates: Any) -> dict[str, Any]:
        allowed = {"receipt_json", "error_code", "executed_at"}
        assignments = ["status=%s", "updated_at=UTC_TIMESTAMP(6)"]
        args: list[Any] = [to_status]
        for key, value in updates.items():
            if key not in allowed:
                continue
            assignments.append(f"{key}=%s")
            args.append(json.dumps(value, ensure_ascii=False) if key == "receipt_json" else value)
        args.extend([action_id, user_id, session_id, from_status])
        ok, affected = self._require().execute_update(
            f"UPDATE cs_governed_actions SET {','.join(assignments)} WHERE action_id=%s AND user_id=%s AND session_id=%s AND status=%s",
            tuple(args),
        )
        if not ok or int(affected) != 1:
            raise RequestConflictError()
        ok, row = self._require().execute_query("SELECT * FROM cs_governed_actions WHERE action_id=%s AND user_id=%s AND session_id=%s", (action_id, user_id, session_id), fetch_one=True)
        if not ok or not row:
            raise StorageOperationError()
        return _action_row(row)

    def execute_action(self, tool_name: str, arguments: dict[str, Any], user_id: str, action_id: str, idempotency_key: str) -> dict[str, Any]:
        client = self._require()
        ok, prior = client.execute_query("SELECT receipt_json FROM cs_tool_call_receipts WHERE idempotency_key=%s LIMIT 1", (idempotency_key,), fetch_one=True)
        if ok and prior:
            value = prior.get("receipt_json")
            return json.loads(value) if isinstance(value, str) else value

        def operation(cursor):
            receipt = self._execute_with_cursor(cursor, tool_name, arguments, user_id, action_id)
            cursor.execute(
                """INSERT INTO cs_tool_call_receipts
                (receipt_id,action_id,user_id,tool_name,idempotency_key,status,receipt_json,created_at)
                VALUES (%s,%s,%s,%s,%s,'succeeded',%s,UTC_TIMESTAMP(6))""",
                (uuid4().hex[:26], action_id, user_id, tool_name, idempotency_key, json.dumps(receipt, ensure_ascii=False, default=str)),
            )
            return receipt

        ok, result = client.execute_in_transaction(operation)
        if not ok:
            raise StorageOperationError()
        return result

    def list_indeterminate(self,limit: int=20) -> list[dict[str,Any]]:
        ok,rows=self._require().execute_query("SELECT * FROM cs_governed_actions WHERE status='indeterminate' ORDER BY updated_at LIMIT %s",(limit,))
        if not ok: raise StorageOperationError()
        return [_action_row(row) for row in rows]

    def get_receipt(self,idempotency_key: str) -> dict[str,Any] | None:
        ok,row=self._require().execute_query("SELECT receipt_json FROM cs_tool_call_receipts WHERE idempotency_key=%s",(idempotency_key,),fetch_one=True)
        if not ok: raise StorageOperationError()
        if not row: return None
        value=row["receipt_json"]
        return json.loads(value) if isinstance(value,str) else _normalize(value)

    def _execute_with_cursor(self, cursor, tool_name: str, args: dict[str, Any], user_id: str, action_id: str) -> dict[str, Any]:
        if tool_name.startswith("telecom_"):
            cursor.execute("SELECT l.* FROM tc_lines l JOIN tc_accounts a ON a.account_id=l.account_id WHERE l.line_id=%s AND a.user_id=%s FOR UPDATE", (args["line_id"], user_id))
            row = cursor.fetchone()
            if not row or int(row["version"]) != int(args["expected_version"]):
                raise RequestConflictError()
            before = int(row["version"])
            if tool_name == "telecom_change_plan":
                cursor.execute("SELECT plan_id,name FROM tc_plans WHERE plan_id=%s AND status='active'", (args["plan_id"],)); plan = cursor.fetchone()
                if not plan: raise RequestConflictError()
                cursor.execute("UPDATE tc_lines SET current_plan_id=%s,last_plan_change_at=UTC_TIMESTAMP(6),version=version+1,updated_at=UTC_TIMESTAMP(6) WHERE line_id=%s", (args["plan_id"], args["line_id"]))
                cursor.execute("INSERT INTO tc_plan_change_history(change_id,line_id,old_plan_id,new_plan_id,action_id,status,created_at) VALUES (%s,%s,%s,%s,%s,'applied',UTC_TIMESTAMP(6))",(uuid4().hex[:26],args["line_id"],row["current_plan_id"],args["plan_id"],action_id))
                summary = {"plan_id": args["plan_id"], "plan_name": plan["name"]}
            elif tool_name == "telecom_refuel_data":
                cursor.execute("SELECT * FROM tc_wallets WHERE account_id=%s FOR UPDATE",(row["account_id"],)); wallet=cursor.fetchone(); charge=Decimal(str(args["quoted_price"]))
                if not wallet or Decimal(str(wallet["available_balance"]))<charge: raise RequestConflictError()
                before_balance=Decimal(str(wallet["available_balance"])); after_balance=before_balance-charge
                cursor.execute("UPDATE tc_wallets SET available_balance=%s,version=version+1,updated_at=UTC_TIMESTAMP(6) WHERE wallet_id=%s",(after_balance,wallet["wallet_id"]))
                cursor.execute("INSERT INTO tc_data_addons (addon_id,line_id,amount_mb,charge_amount,currency,action_id,status,created_at) VALUES (%s,%s,%s,%s,'CNY',%s,'applied',UTC_TIMESTAMP(6))", (uuid4().hex[:26], args["line_id"], args["amount_mb"], args["quoted_price"], action_id))
                cursor.execute("INSERT INTO tc_wallet_ledger(ledger_id,wallet_id,action_id,direction,amount,balance_before,balance_after,entry_type,reference_type,reference_id,created_at) VALUES (%s,%s,%s,'debit',%s,%s,%s,'data_refuel','line',%s,UTC_TIMESTAMP(6))",(uuid4().hex[:26],wallet["wallet_id"],action_id,charge,before_balance,after_balance,args["line_id"]))
                cursor.execute("UPDATE tc_usage_cycles SET refueled_data_mb=refueled_data_mb+%s,version=version+1,updated_at=UTC_TIMESTAMP(6) WHERE line_id=%s AND cycle_end>=UTC_TIMESTAMP(6)", (args["amount_mb"], args["line_id"]))
                cursor.execute("UPDATE tc_lines SET version=version+1,updated_at=UTC_TIMESTAMP(6) WHERE line_id=%s", (args["line_id"],)); summary={"amount_mb":args["amount_mb"],"charged":str(args["quoted_price"])}
            elif tool_name == "telecom_set_roaming":
                cursor.execute("UPDATE tc_lines SET roaming_enabled=%s,version=version+1,updated_at=UTC_TIMESTAMP(6) WHERE line_id=%s", (bool(args["enabled"]), args["line_id"])); summary={"roaming_enabled":bool(args["enabled"])}
                cursor.execute("INSERT INTO tc_roaming_history(roaming_change_id,line_id,enabled,action_id,created_at) VALUES (%s,%s,%s,%s,UTC_TIMESTAMP(6))",(uuid4().hex[:26],args["line_id"],bool(args["enabled"]),action_id))
            else: raise StorageOperationError()
            return {"status":"succeeded","resource_type":"line","resource_id":args["line_id"],"version_before":before,"version_after":before+1,"summary":summary}

        if tool_name == "retail_create_address":
            address_id = uuid4().hex[:26]
            set_default = bool(args["set_default"])
            if set_default:
                cursor.execute(
                    "UPDATE cs_user_addresses SET is_default=FALSE,version=version+1,"
                    "updated_at=UTC_TIMESTAMP(6) WHERE user_id=%s AND status='active' AND is_default=TRUE",
                    (user_id,),
                )
            cursor.execute(
                """INSERT INTO cs_user_addresses
                (address_id,user_id,label,recipient_cipher,phone_cipher,province,city,district,
                 detail_cipher,postal_code,is_default,status,version,created_at,updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'active',1,UTC_TIMESTAMP(6),UTC_TIMESTAMP(6))""",
                (
                    address_id,
                    user_id,
                    args.get("label") or "默认收货地址",
                    encrypt_pii(args["recipient"]),
                    encrypt_pii(args["phone"]),
                    args["province"],
                    args["city"],
                    args["district"],
                    encrypt_pii(args["detail"]),
                    args.get("postal_code"),
                    set_default,
                ),
            )
            return {
                "status": "succeeded",
                "resource_type": "address",
                "resource_id": address_id,
                "version_before": 0,
                "version_after": 1,
                "summary": {
                    "label": args.get("label") or "默认收货地址",
                    "recipient": args["recipient"],
                    "phone": args["phone"],
                    "full_address": (
                        f"{args['province']}{args['city']}{args['district']}{args['detail']}"
                    ),
                    "is_default": set_default,
                },
            }

        cursor.execute("SELECT * FROM rt_orders WHERE order_id=%s AND user_id=%s FOR UPDATE", (args.get("order_id"), user_id))
        order = cursor.fetchone()
        if tool_name == "retail_set_default_address":
            cursor.execute("SELECT * FROM cs_user_addresses WHERE address_id=%s AND user_id=%s AND status='active' FOR UPDATE", (args["address_id"], user_id)); address=cursor.fetchone()
            if not address or int(address["version"]) != int(args["expected_version"]): raise RequestConflictError()
            before=int(address["version"]); cursor.execute("UPDATE cs_user_addresses SET is_default=(address_id=%s),version=version+1,updated_at=UTC_TIMESTAMP(6) WHERE user_id=%s AND status='active'", (args["address_id"],user_id))
            return {"status":"succeeded","resource_type":"address","resource_id":args["address_id"],"version_before":before,"version_after":before+1,"summary":{"is_default":True}}
        if not order or int(order["version"]) != int(args["expected_version"]): raise RequestConflictError()
        before=int(order["version"]); status=str(order["status"])
        allowed_statuses={
            "retail_cancel_order":{"pending"},"retail_update_order_address":{"pending"},
            "retail_update_order_payment":{"pending"},"retail_update_order_items":{"pending"},
            "retail_request_return":{"delivered"},"retail_request_exchange":{"delivered"},
            "retail_price_adjustment_refund":{"delivered"},
        }
        if status not in allowed_statuses.get(tool_name,set()): raise RequestConflictError()
        if tool_name == "retail_cancel_order": cursor.execute("UPDATE rt_orders SET status='cancelled',cancel_reason=%s,version=version+1,updated_at=UTC_TIMESTAMP(6) WHERE order_id=%s",(args["reason"],args["order_id"])); summary={"status":"cancelled","reason":args["reason"]}
        elif tool_name in {"retail_update_order_address","retail_update_order_payment","retail_update_order_items"}:
            change_type=tool_name.rsplit("_",1)[-1]
            if tool_name=="retail_update_order_address":
                cursor.execute("SELECT address_id,label,province,city,district,postal_code,version FROM cs_user_addresses WHERE address_id=%s AND user_id=%s AND status='active'",(args["address_id"],user_id)); selected=cursor.fetchone()
                if not selected: raise RequestConflictError()
                cursor.execute("INSERT INTO rt_order_addresses(order_address_id,order_id,address_type,source_address_id,address_snapshot_json,version,created_at,updated_at) VALUES (%s,%s,'shipping',%s,%s,1,UTC_TIMESTAMP(6),UTC_TIMESTAMP(6)) ON DUPLICATE KEY UPDATE source_address_id=VALUES(source_address_id),address_snapshot_json=VALUES(address_snapshot_json),version=version+1,updated_at=UTC_TIMESTAMP(6)",(uuid4().hex[:26],args["order_id"],args["address_id"],json.dumps(_normalize(selected),ensure_ascii=False)))
            elif tool_name=="retail_update_order_payment":
                cursor.execute("SELECT payment_method_id FROM cs_payment_methods WHERE payment_method_id=%s AND user_id=%s AND status='active'",(args["payment_method_id"],user_id))
                if not cursor.fetchone(): raise RequestConflictError()
                cursor.execute("UPDATE rt_order_payments SET payment_method_id=%s,version=version+1,updated_at=UTC_TIMESTAMP(6) WHERE order_id=%s AND status IN ('pending','authorized')",(args["payment_method_id"],args["order_id"]))
            else:
                for requested in args["items"]:
                    cursor.execute("SELECT v.variant_id,i.available_qty FROM rt_product_variants v JOIN rt_inventory i ON i.variant_id=v.variant_id WHERE v.variant_id=%s AND v.status='active' FOR UPDATE",(requested["variant_id"],)); variant=cursor.fetchone()
                    if not variant or int(variant["available_qty"])<int(requested["quantity"]): raise RequestConflictError()
            cursor.execute("INSERT INTO rt_order_changes (change_id,order_id,user_id,change_type,change_summary_json,price_difference,action_id,status,created_at) VALUES (%s,%s,%s,%s,%s,0,%s,'applied',UTC_TIMESTAMP(6))",(uuid4().hex[:26],args["order_id"],user_id,change_type,json.dumps(args,ensure_ascii=False),action_id)); cursor.execute("UPDATE rt_orders SET items_modified=%s,version=version+1,updated_at=UTC_TIMESTAMP(6) WHERE order_id=%s",(tool_name.endswith("items"),args["order_id"])); summary={"change_type":change_type}
        elif tool_name in {"retail_request_return","retail_request_exchange"}:
            kind="return" if tool_name.endswith("return") else "exchange"; table="rt_returns" if kind=="return" else "rt_exchanges"; key=f"{kind}_id"; rid=uuid4().hex[:26]; cursor.execute(f"INSERT INTO {table} ({key},order_id,user_id,reason,status,action_id,version,created_at,updated_at) VALUES (%s,%s,%s,%s,'requested',%s,1,UTC_TIMESTAMP(6),UTC_TIMESTAMP(6))",(rid,args["order_id"],user_id,args["reason"],action_id))
            item_table="rt_return_items" if kind=="return" else "rt_exchange_items"; item_key=f"{kind}_item_id"
            for requested in args["items"]:
                cursor.execute("SELECT quantity,returned_qty,exchanged_qty FROM rt_order_items WHERE order_item_id=%s AND order_id=%s FOR UPDATE",(requested["order_item_id"],args["order_id"])); order_item=cursor.fetchone()
                used=int(order_item.get("returned_qty",0))+int(order_item.get("exchanged_qty",0)) if order_item else 0
                if not order_item or int(requested["quantity"])+used>int(order_item["quantity"]): raise RequestConflictError()
                cursor.execute(f"INSERT INTO {item_table}({item_key},{key},order_item_id,quantity,created_at) VALUES (%s,%s,%s,%s,UTC_TIMESTAMP(6))",(uuid4().hex[:26],rid,requested["order_item_id"],requested["quantity"]))
            cursor.execute(f"UPDATE rt_orders SET status='{kind}_requested',version=version+1,updated_at=UTC_TIMESTAMP(6) WHERE order_id=%s",(args["order_id"],)); summary={key:rid,"items":args["items"]}
        elif tool_name == "retail_price_adjustment_refund":
            rid=uuid4().hex[:26]; cursor.execute("INSERT INTO rt_refunds (refund_id,order_id,user_id,amount,currency,method,status,action_id,version,created_at,updated_at) VALUES (%s,%s,%s,%s,'CNY',%s,'pending',%s,1,UTC_TIMESTAMP(6),UTC_TIMESTAMP(6))",(rid,args["order_id"],user_id,args["amount"],args["method"],action_id)); cursor.execute("UPDATE rt_orders SET version=version+1,updated_at=UTC_TIMESTAMP(6) WHERE order_id=%s",(args["order_id"],)); summary={"refund_id":rid,"amount":str(args["amount"]),"method":args["method"]}
        else: raise StorageOperationError()
        return {"status":"succeeded","resource_type":"order","resource_id":args["order_id"],"version_before":before,"version_after":before+1,"summary":summary}


def _normalize(value: Any) -> Any:
    if isinstance(value, dict): return {key:_normalize(item) for key,item in value.items()}
    if isinstance(value, list): return [_normalize(item) for item in value]
    if isinstance(value, Decimal): return str(value)
    if isinstance(value, datetime): return value.replace(tzinfo=timezone.utc).isoformat()
    if isinstance(value, (bytes, bytearray)): return "[encrypted]"
    return value


def _naive(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _action_row(row: dict[str, Any]) -> dict[str, Any]:
    item={key:_normalize(row.get(key)) for key in ("action_id","user_id","session_id","turn_id","tool_name","tool_version","skill_name","skill_version","arguments_digest","impact_summary","status","idempotency_key","resource_version","error_code")}
    for source,target in (("arguments_json","arguments"),("receipt_json","receipt")):
        value=row.get(source)
        if source == "arguments_json":
            item[target] = _decode_sensitive_json(value)
        else:
            item[target]=json.loads(value) if isinstance(value,str) else _normalize(value)
    for key in ("created_at","updated_at","expires_at","executed_at"):
        value=row.get(key)
        if isinstance(value,datetime): item[key]=value.replace(tzinfo=timezone.utc)
    return item


def _encode_sensitive_json(value: dict[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    token = encrypt_pii(raw).decode("ascii")
    return json.dumps({"_encrypted_payload": token}, separators=(",", ":"))


def _decode_sensitive_json(value: Any) -> dict[str, Any]:
    parsed = json.loads(value) if isinstance(value, str) else _normalize(value)
    if isinstance(parsed, dict) and set(parsed) == {"_encrypted_payload"}:
        plaintext = decrypt_pii(parsed["_encrypted_payload"])
        if plaintext is None:
            raise StorageOperationError()
        decoded = json.loads(plaintext)
        if not isinstance(decoded, dict):
            raise StorageOperationError()
        return decoded
    if not isinstance(parsed, dict):
        raise StorageOperationError()
    return parsed
