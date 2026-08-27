"""Seed realistic synthetic telecom and retail demo data.

The seed is additive and repeatable: fixed identifiers plus no-op duplicate updates
prevent a second run from resetting business state changed through governed actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pymysql
from pymysql.cursors import DictCursor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pkg.config.settings import get_settings  # noqa: E402
from pkg.security import encrypt_pii  # noqa: E402


SEED_COUNT = 20

NAMES = [
    "张伟", "王芳", "李娜", "刘洋", "陈晨", "杨帆", "赵敏", "黄杰", "周婷", "吴磊",
    "徐静", "孙浩", "胡悦", "朱琳", "高远", "林欣", "何俊", "郭佳", "马超", "罗雪",
]

LOCATIONS = [
    ("北京市", "北京市", "朝阳区", "100020"), ("上海市", "上海市", "浦东新区", "200120"),
    ("广东省", "广州市", "天河区", "510630"), ("广东省", "深圳市", "南山区", "518052"),
    ("浙江省", "杭州市", "西湖区", "310013"), ("江苏省", "南京市", "鼓楼区", "210009"),
    ("四川省", "成都市", "武侯区", "610041"), ("湖北省", "武汉市", "洪山区", "430070"),
    ("陕西省", "西安市", "雁塔区", "710061"), ("福建省", "厦门市", "思明区", "361001"),
    ("山东省", "青岛市", "市南区", "266001"), ("湖南省", "长沙市", "岳麓区", "410013"),
    ("河南省", "郑州市", "金水区", "450003"), ("重庆市", "重庆市", "渝中区", "400010"),
    ("天津市", "天津市", "和平区", "300041"), ("辽宁省", "大连市", "中山区", "116001"),
    ("安徽省", "合肥市", "蜀山区", "230031"), ("江西省", "南昌市", "红谷滩区", "330038"),
    ("云南省", "昆明市", "五华区", "650032"), ("广西壮族自治区", "南宁市", "青秀区", "530022"),
]

PRODUCTS = [
    ("5G随身WiFi Pro", "网络设备", "支持5G双模和多设备共享的便携热点", Decimal("599.00")),
    ("千兆双频路由器", "网络设备", "Wi-Fi 6千兆路由器，适合中小户型", Decimal("299.00")),
    ("智能儿童手表", "智能穿戴", "支持通话、定位和校园模式", Decimal("699.00")),
    ("运动智能手环", "智能穿戴", "心率、睡眠和运动记录", Decimal("249.00")),
    ("降噪蓝牙耳机", "影音配件", "主动降噪与通话降噪", Decimal("399.00")),
    ("Type-C快充套装", "手机配件", "65W充电器和1.5米数据线", Decimal("129.00")),
    ("磁吸无线充电器", "手机配件", "15W无线快充，带过温保护", Decimal("159.00")),
    ("高清网络摄像头", "智能家居", "支持夜视和移动侦测", Decimal("329.00")),
    ("智能门铃", "智能家居", "远程可视对讲和云端提醒", Decimal("499.00")),
    ("全屋WiFi子路由", "网络设备", "Mesh组网扩展覆盖", Decimal("369.00")),
    ("便携蓝牙音箱", "影音配件", "防水设计，续航12小时", Decimal("219.00")),
    ("手机稳定器", "影像配件", "三轴防抖和智能跟拍", Decimal("459.00")),
    ("高清投屏器", "影音配件", "支持4K无线投屏", Decimal("269.00")),
    ("智能插座套装", "智能家居", "远程控制和用电统计，两只装", Decimal("119.00")),
    ("移动电源20000mAh", "手机配件", "双向快充并支持多口输出", Decimal("189.00")),
    ("USB-C扩展坞", "电脑配件", "HDMI、网口和多接口扩展", Decimal("239.00")),
    ("折叠手机支架", "手机配件", "铝合金桌面支架", Decimal("59.00")),
    ("4G老人手机", "手机终端", "大字体、大音量和一键呼叫", Decimal("399.00")),
    ("入门5G手机", "手机终端", "大电池双卡5G智能手机", Decimal("1299.00")),
    ("旗舰5G手机", "手机终端", "高性能影像旗舰手机", Decimal("4999.00")),
]


def _id(prefix: str, index: int) -> str:
    return f"{prefix}{index:0{26 - len(prefix)}d}"


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def _opaque(value: str) -> bytes:
    return encrypt_pii(value)


TABLE_ORDER = (
    "cs_users", "cs_sessions", "cs_user_addresses", "cs_payment_methods",
    "tc_accounts", "tc_wallets", "tc_plans", "tc_lines", "tc_usage_cycles",
    "rt_products", "rt_product_variants", "rt_inventory", "rt_orders",
    "rt_order_items", "rt_order_addresses", "rt_order_payments",
)


def build_seed_rows(now: datetime | None = None, count: int = SEED_COUNT) -> dict[str, list[tuple[Any, ...]]]:
    if count < 1 or count > SEED_COUNT:
        raise ValueError(f"count must be between 1 and {SEED_COUNT}")
    now = (now or datetime.now(timezone.utc)).replace(tzinfo=None, microsecond=0)
    cycle_start = now.replace(day=1, hour=0, minute=0, second=0)
    next_month = (cycle_start.replace(day=28) + timedelta(days=4)).replace(day=1)
    cycle_end = next_month - timedelta(microseconds=1)
    rows: dict[str, list[tuple[Any, ...]]] = {name: [] for name in TABLE_ORDER}

    for i in range(1, count + 1):
        user_id = f"demo_user_{i:03d}"
        phone = f"1380000{i:04d}"
        email = f"demo.user{i:03d}@example.test"
        province, city, district, postal_code = LOCATIONS[i - 1]
        user_status = "active" if i <= 18 else ("suspended" if i == 19 else "closed")
        rows["cs_users"].append((
            user_id, "demo", f"demo-subject-{i:03d}", user_status, NAMES[i - 1],
            _opaque(email), _digest(email.lower()), _opaque(phone), _digest(phone), 1,
            now - timedelta(days=200 - i), now, None,
        ))
        rows["cs_sessions"].append((
            f"demo_session_{i:03d}", user_id, f"{NAMES[i - 1]}的演示会话", "general",
            False, 0, "active", 1, now, now,
        ))
        rows["cs_user_addresses"].append((
            _id("adr", i), user_id, "默认收货地址", _opaque(NAMES[i - 1]), _opaque(phone),
            province, city, district, _opaque(f"演示路{i}号{i}单元{i}室"), postal_code, True,
            "active", 1, now, now,
        ))
        payment_type = ("alipay", "wechat_pay", "bank_card")[i % 3]
        rows["cs_payment_methods"].append((
            _id("pay", i), user_id, payment_type, _opaque(f"provider-token-{i:03d}"),
            f"{6000 + i:04d}"[-4:], None, "CNY", "active", 1, now, now,
        ))
        rows["tc_accounts"].append((
            _id("acc", i), user_id, f"CMCC-DEMO-{20260000 + i}",
            "active" if i <= 18 else "suspended", 1, now - timedelta(days=180 - i), now,
        ))
        rows["tc_wallets"].append((_id("wal", i), _id("acc", i), Decimal(50 + i * 12), "CNY", 1, now))

        data_gb = [5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200, 300, 500, 999, 999][i - 1]
        price = Decimal([19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 129, 149, 169, 199, 229, 259, 299, 399, 499, 599][i - 1])
        unlimited = i >= 19
        voice_minutes = [100, 100, 150, 200, 200, 300, 300, 400, 500, 600, 700, 800, 1000, 1200, 1500, 1800, 2000, 3000, 0, 0][i - 1]
        plan_status = "active" if i <= 18 else "retired"
        rows["tc_plans"].append((
            _id("pln", i), f"CMCC-P{i:03d}", 1, f"畅享{data_gb if not unlimited else '无限'}G-{int(price)}元套餐",
            f"包含{'不限量' if unlimited else str(data_gb) + 'GB'}国内流量和{voice_minutes if voice_minutes else '不限量'}分钟国内语音",
            price, "CNY", data_gb * 1024, unlimited, voice_minutes, unlimited,
            Decimal("0.15"), Decimal("5.00" if i >= 10 else "10.00"), 50 * 1024,
            i >= 6, plan_status, now - timedelta(days=365), None, now, now,
        ))
        rows["tc_lines"].append((
            _id("lin", i), _id("acc", i), _opaque(phone), _digest(phone),
            "active" if i <= 18 else "suspended", _id("pln", ((i + 3) % 18) + 1), i % 4 == 0,
            now + timedelta(days=90 + i), now - timedelta(days=60 + i), 1,
            now - timedelta(days=180 - i), now,
        ))
        included_mb = [5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200, 300, 500][((i + 3) % 18)] * 1024
        used_mb = min(included_mb + 2048, 1800 + i * 2350)
        rows["tc_usage_cycles"].append((
            _id("use", i), _id("lin", i), cycle_start, cycle_end, included_mb, used_mb,
            1024 if i % 6 == 0 else 0, 100 + i * 50, 35 + i * 37, 8 + i * 3,
            1 if i % 6 == 0 else 0, 1, now,
        ))

        product_name, category, description, product_price = PRODUCTS[i - 1]
        product_status = "active" if i <= 19 else "discontinued"
        rows["rt_products"].append((
            _id("prd", i), f"DEMO-PROD-{i:03d}", product_name, category, description,
            product_status, 1, now - timedelta(days=120 - i), now,
        ))
        attributes = json.dumps({"颜色": ("曜石黑", "云杉绿", "星河银")[i % 3], "版本": "标准版"}, ensure_ascii=False)
        rows["rt_product_variants"].append((
            _id("var", i), _id("prd", i), f"DEMO-SKU-{i:03d}", attributes,
            product_price, "CNY", product_status, 1, now - timedelta(days=120 - i), now,
        ))
        rows["rt_inventory"].append((_id("var", i), 15 + i * 4, i % 5, 1, now))

        quantity = 2 if i % 7 == 0 else 1
        item_total = product_price * quantity
        shipping = Decimal("0.00") if item_total >= 199 else Decimal("10.00")
        discount = Decimal("20.00") if i % 5 == 0 else Decimal("0.00")
        grand_total = item_total + shipping - discount
        order_status = ("pending", "paid", "shipped", "delivered", "cancelled")[(i - 1) % 5]
        placed_at = now - timedelta(days=i * 2)
        rows["rt_orders"].append((
            _id("ord", i), f"DEMO{now:%Y%m}{i:06d}", user_id, order_status,
            item_total, shipping, discount, grand_total, "CNY",
            "用户改变购买计划" if order_status == "cancelled" else None, False, 1,
            placed_at, placed_at, now,
        ))
        rows["rt_order_items"].append((
            _id("itm", i), _id("ord", i), _id("prd", i), _id("var", i), f"DEMO-SKU-{i:03d}",
            product_name, attributes, product_price, quantity, 0, 0, "active", 1,
        ))
        address_snapshot = json.dumps({
            "recipient": NAMES[i - 1], "phone_masked": f"138****{phone[-4:]}",
            "province": province, "city": city, "district": district,
            "detail": f"演示路{i}号{i}单元{i}室", "postal_code": postal_code,
        }, ensure_ascii=False)
        rows["rt_order_addresses"].append((
            _id("oad", i), _id("ord", i), "shipping", _id("adr", i),
            address_snapshot, 1, placed_at, now,
        ))
        payment_status = "refunded" if order_status == "cancelled" else ("authorized" if order_status == "pending" else "paid")
        rows["rt_order_payments"].append((
            _id("opm", i), _id("ord", i), _id("pay", i), grand_total, "CNY",
            payment_status, 1, placed_at, now,
        ))

    return rows


INSERT_SQL = {
    "cs_users": "INSERT INTO cs_users (user_id,tenant_id,auth_subject,status,display_name,email_cipher,email_hash,phone_cipher,phone_hash,version,created_at,updated_at,deleted_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE display_name=VALUES(display_name),email_cipher=VALUES(email_cipher),email_hash=VALUES(email_hash),phone_cipher=VALUES(phone_cipher),phone_hash=VALUES(phone_hash)",
    "cs_sessions": "INSERT INTO cs_sessions (session_id,user_id,title,agent_id,favorite,message_count,status,version,created_at,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE session_id=VALUES(session_id)",
    "cs_user_addresses": "INSERT INTO cs_user_addresses (address_id,user_id,label,recipient_cipher,phone_cipher,province,city,district,detail_cipher,postal_code,is_default,status,version,created_at,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE label=VALUES(label),recipient_cipher=VALUES(recipient_cipher),phone_cipher=VALUES(phone_cipher),province=VALUES(province),city=VALUES(city),district=VALUES(district),detail_cipher=VALUES(detail_cipher),postal_code=VALUES(postal_code)",
    "cs_payment_methods": "INSERT INTO cs_payment_methods (payment_method_id,user_id,type,provider_token_cipher,last4,gift_card_balance,currency,status,version,created_at,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE payment_method_id=VALUES(payment_method_id)",
    "tc_accounts": "INSERT INTO tc_accounts (account_id,user_id,account_no,status,version,created_at,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE account_id=VALUES(account_id)",
    "tc_wallets": "INSERT INTO tc_wallets (wallet_id,account_id,available_balance,currency,version,updated_at) VALUES (%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE wallet_id=VALUES(wallet_id)",
    "tc_plans": "INSERT INTO tc_plans (plan_id,plan_code,version_no,name,description,monthly_price,currency,data_limit_mb,data_unlimited,included_voice_minutes,voice_unlimited,voice_overage_price_per_minute,refuel_price_per_gb,max_refuel_mb_per_cycle,roaming_supported,status,effective_from,effective_to,created_at,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE plan_id=VALUES(plan_id)",
    "tc_lines": "INSERT INTO tc_lines (line_id,account_id,msisdn_cipher,msisdn_hash,status,current_plan_id,roaming_enabled,contract_end_at,last_plan_change_at,version,created_at,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE msisdn_cipher=VALUES(msisdn_cipher),msisdn_hash=VALUES(msisdn_hash)",
    "tc_usage_cycles": "INSERT INTO tc_usage_cycles (usage_id,line_id,cycle_start,cycle_end,included_data_mb,used_data_mb,refueled_data_mb,included_voice_minutes,used_voice_minutes,outgoing_call_count,refuel_count,version,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE usage_id=VALUES(usage_id)",
    "rt_products": "INSERT INTO rt_products (product_id,product_code,name,category,description,status,version,created_at,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE product_id=VALUES(product_id)",
    "rt_product_variants": "INSERT INTO rt_product_variants (variant_id,product_id,sku,attributes_json,price,currency,status,version,created_at,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE variant_id=VALUES(variant_id)",
    "rt_inventory": "INSERT INTO rt_inventory (variant_id,available_qty,reserved_qty,version,updated_at) VALUES (%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE variant_id=VALUES(variant_id)",
    "rt_orders": "INSERT INTO rt_orders (order_id,order_no,user_id,status,item_total,shipping_fee,discount_total,grand_total,currency,cancel_reason,items_modified,version,placed_at,created_at,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE order_id=VALUES(order_id)",
    "rt_order_items": "INSERT INTO rt_order_items (order_item_id,order_id,product_id,variant_id,sku_snapshot,name_snapshot,attributes_snapshot_json,unit_price,quantity,returned_qty,exchanged_qty,item_status,version) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE order_item_id=VALUES(order_item_id)",
    "rt_order_addresses": "INSERT INTO rt_order_addresses (order_address_id,order_id,address_type,source_address_id,address_snapshot_json,version,created_at,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE order_address_id=VALUES(order_address_id)",
    "rt_order_payments": "INSERT INTO rt_order_payments (payment_id,order_id,payment_method_id,amount,currency,status,version,created_at,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE payment_id=VALUES(payment_id)",
}


def apply_seed(rows: dict[str, list[tuple[Any, ...]]]) -> dict[str, int]:
    settings = get_settings()
    connection = pymysql.connect(
        host=settings.db_host, port=settings.db_port, user=settings.db_user,
        password=settings.db_password, database=settings.db_name, charset="utf8mb4",
        cursorclass=DictCursor, connect_timeout=settings.db_connect_timeout_seconds,
        read_timeout=settings.db_read_timeout_seconds, write_timeout=settings.db_write_timeout_seconds,
    )
    try:
        with connection.cursor() as cursor:
            _reconcile_session_schema(cursor)
            connection.commit()
            for table in TABLE_ORDER:
                try:
                    cursor.executemany(INSERT_SQL[table], rows[table])
                except Exception as exc:
                    raise RuntimeError(f"failed to seed table {table}: {exc}") from exc
            connection.commit()
            counts: dict[str, int] = {}
            for table in TABLE_ORDER:
                cursor.execute(f"SELECT COUNT(*) AS count FROM {table}")
                counts[table] = int(cursor.fetchone()["count"])
            return counts
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _reconcile_session_schema(cursor: Any) -> None:
    columns = {
        "title": "ALTER TABLE cs_sessions ADD COLUMN title VARCHAR(255) NOT NULL DEFAULT '新会话' AFTER user_id",
        "agent_id": "ALTER TABLE cs_sessions ADD COLUMN agent_id VARCHAR(64) NOT NULL DEFAULT 'general' AFTER title",
        "favorite": "ALTER TABLE cs_sessions ADD COLUMN favorite TINYINT(1) NOT NULL DEFAULT 0 AFTER agent_id",
        "message_count": "ALTER TABLE cs_sessions ADD COLUMN message_count INT NOT NULL DEFAULT 0 AFTER favorite",
        "status": "ALTER TABLE cs_sessions ADD COLUMN status VARCHAR(32) NOT NULL DEFAULT 'active' AFTER message_count",
        "version": "ALTER TABLE cs_sessions ADD COLUMN version BIGINT UNSIGNED NOT NULL DEFAULT 1 AFTER status",
    }
    cursor.execute("SHOW COLUMNS FROM cs_sessions")
    existing = {str(row["Field"]) for row in cursor.fetchall()}
    for column, ddl in columns.items():
        if column not in existing:
            cursor.execute(ddl)


def inspect_schema() -> dict[str, list[str]]:
    settings = get_settings()
    connection = pymysql.connect(
        host=settings.db_host, port=settings.db_port, user=settings.db_user,
        password=settings.db_password, database=settings.db_name, charset="utf8mb4",
        cursorclass=DictCursor, connect_timeout=settings.db_connect_timeout_seconds,
    )
    try:
        with connection.cursor() as cursor:
            result: dict[str, list[str]] = {}
            for table in TABLE_ORDER:
                cursor.execute(f"SHOW COLUMNS FROM {table}")
                result[table] = [str(row["Field"]) for row in cursor.fetchall()]
            return result
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed synthetic telecom/retail business data")
    parser.add_argument("--count", type=int, default=SEED_COUNT, help="rows per primary table (1-20)")
    parser.add_argument("--apply", action="store_true", help="write to the configured MySQL database")
    parser.add_argument("--inspect", action="store_true", help="print actual columns of target tables")
    args = parser.parse_args()
    if args.inspect:
        print(json.dumps(inspect_schema(), ensure_ascii=False, indent=2))
        return 0
    rows = build_seed_rows(count=args.count)
    if not args.apply:
        print(json.dumps({table: len(values) for table, values in rows.items()}, ensure_ascii=False, indent=2))
        print("Dry run only. Add --apply to write the configured database.")
        return 0
    counts = apply_seed(rows)
    print(json.dumps({"seeded": {table: len(rows[table]) for table in TABLE_ORDER}, "database_counts": counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
