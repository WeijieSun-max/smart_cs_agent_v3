from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any, Protocol

from evaluation.schema import InitialState


_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")


class QueryClient(Protocol):
    def execute_query(
        self,
        sql: str,
        args: Any = None,
        fetch_one: bool = False,
    ) -> tuple[bool, Any]: ...


EnvironmentResetter = Callable[[QueryClient, InitialState], None]


class MySQLEvaluationEnvironment:
    """A fail-closed adapter for a dedicated MySQL evaluation database.

    It never creates, drops, truncates, or resets data itself. A caller must supply
    a scoped reset function, and both the configured and connected database names
    must use the explicit evaluation prefix.
    """

    DEFAULT_SNAPSHOT_TABLES = (
        "tc_lines",
        "tc_plan_change_history",
        "rt_orders",
        "rt_order_changes",
        "cs_governed_actions",
        "cs_tool_call_receipts",
    )

    def __init__(
        self,
        client: QueryClient,
        database_name: str,
        *,
        resetter: EnvironmentResetter | None = None,
        snapshot_tables: Sequence[str] = DEFAULT_SNAPSHOT_TABLES,
        required_prefix: str = "smart_cs_eval_",
    ) -> None:
        if not required_prefix or not database_name.startswith(required_prefix):
            raise ValueError(
                f"evaluation database must start with the isolated prefix {required_prefix!r}"
            )
        if not _IDENTIFIER.fullmatch(database_name):
            raise ValueError("evaluation database name contains unsafe characters")
        tables = tuple(snapshot_tables)
        if any(not _IDENTIFIER.fullmatch(table) for table in tables):
            raise ValueError("snapshot table name contains unsafe characters")
        self._client = client
        self.database_name = database_name
        self._resetter = resetter
        self._snapshot_tables = tables

    def assert_isolated(self) -> None:
        ok, row = self._client.execute_query("SELECT DATABASE() AS database_name", fetch_one=True)
        connected = row.get("database_name") if ok and isinstance(row, dict) else None
        if connected != self.database_name:
            raise RuntimeError(
                "MySQL evaluation connection is not bound to the declared isolated database"
            )

    def reset(self, initial_state: InitialState) -> None:
        self.assert_isolated()
        if self._resetter is None:
            raise RuntimeError("MySQL evaluation reset requires an explicit scoped resetter")
        self._resetter(self._client, initial_state)

    def snapshot(self) -> dict[str, Any]:
        self.assert_isolated()
        output: dict[str, Any] = {}
        for table in self._snapshot_tables:
            ok, rows = self._client.execute_query(f"SELECT * FROM `{table}`")
            if not ok:
                raise RuntimeError(f"failed to snapshot evaluation table: {table}")
            output[table] = deepcopy(rows)
        return output
