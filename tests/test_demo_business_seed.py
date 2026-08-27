from __future__ import annotations

from datetime import datetime, timezone

from scripts.seed_demo_business_data import build_seed_rows


def test_demo_seed_builds_three_completed_cycles_and_one_current_cycle_per_line() -> None:
    now = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)

    rows = build_seed_rows(now=now, count=2)
    usage_rows = rows["tc_usage_cycles"]

    assert len(usage_rows) == 8
    for line_index in (1, 2):
        line_id = f"lin{line_index:023d}"
        cycles = sorted(
            (row for row in usage_rows if row[1] == line_id),
            key=lambda row: row[2],
        )
        assert len(cycles) == 4
        assert len({row[0] for row in cycles}) == 4
        assert all(len(row[0]) == 26 for row in cycles)
        assert all(row[3] < now.replace(tzinfo=None) for row in cycles[:3])
        assert cycles[3][2] <= now.replace(tzinfo=None) <= cycles[3][3]


def test_first_demo_user_has_enough_history_for_a_data_driven_recommendation() -> None:
    now = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)

    rows = build_seed_rows(now=now, count=1)
    completed = sorted(rows["tc_usage_cycles"], key=lambda row: row[2])[:3]

    assert all(row[5] > row[4] for row in completed)
    assert all(row[10] == 1 for row in completed)


def test_demo_seed_builds_mixed_retail_order_history_for_each_user() -> None:
    now = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)

    rows = build_seed_rows(now=now, count=2)
    user_orders = [row for row in rows["rt_orders"] if row[2] == "demo_user_001"]
    user_order_ids = {row[0] for row in user_orders}
    user_items = [row for row in rows["rt_order_items"] if row[1] in user_order_ids]

    assert len(rows["rt_orders"]) == 8
    assert len(user_orders) == 4
    assert {row[3] for row in user_orders} == {
        "pending",
        "processed",
        "delivered",
        "cancelled",
    }
    assert len(user_items) == 4
    assert len({row[4] for row in user_items}) == 2
