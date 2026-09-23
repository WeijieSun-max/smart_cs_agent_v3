from __future__ import annotations

from typing import Any, Callable

import pymysql
from dbutils.pooled_db import PooledDB
from pymysql.cursors import DictCursor

from pkg.config.settings import get_settings
from pkg.log.logger import get_logger
from pkg.telemetry import normalize_error, safe_error_status

logger = get_logger()
_mysql_client: MySQLClient | None = None


class MySQLClient:
    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        db_name: str,
        *,
        max_connections: int = 10,
        min_cached: int = 1,
        max_cached: int = 5,
        blocking: bool = False,
        connect_timeout: int = 5,
        read_timeout: int = 10,
        write_timeout: int = 10,
    ) -> None:
        del blocking  # DBUtils cannot provide a bounded blocking wait.
        self._pool = PooledDB(
            creator=pymysql,
            maxconnections=max_connections,
            mincached=min_cached,
            maxcached=max_cached,
            # DBUtils has no bounded pool wait. Failing fast is safer than an
            # unbounded request stall when the pool is exhausted.
            blocking=False,
            ping=1,
            host=host,
            port=port,
            user=user,
            password=password,
            database=db_name,
            charset="utf8mb4",
            cursorclass=DictCursor,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
        )

    def execute_query(self, sql: str, args: Any = None, fetch_one: bool = False) -> tuple[bool, Any]:
        connection = cursor = None
        try:
            connection = self._pool.connection()
            cursor = connection.cursor()
            cursor.execute(sql, args)
            result = cursor.fetchone() if fetch_one else cursor.fetchall()
            return True, result
        except Exception as exc:
            error = normalize_error(exc)
            logger.warning(
                "MySQL query failed error_type={} error_code={}",
                error["error_type"],
                error["error_code"],
            )
            return False, safe_error_status(exc)
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()

    def execute_update(self, sql: str, args: Any = None) -> tuple[bool, Any]:
        connection = cursor = None
        try:
            connection = self._pool.connection()
            cursor = connection.cursor()
            affected_rows = cursor.execute(sql, args)
            connection.commit()
            return True, affected_rows
        except Exception as exc:
            if connection is not None:
                connection.rollback()
            error = normalize_error(exc)
            logger.warning(
                "MySQL update failed error_type={} error_code={}",
                error["error_type"],
                error["error_code"],
            )
            return False, safe_error_status(exc)
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()

    def execute_transaction(self, statements: list[tuple[str, Any]]) -> tuple[bool, Any]:
        connection = cursor = None
        try:
            connection = self._pool.connection()
            cursor = connection.cursor()
            affected_rows = 0
            for sql, args in statements:
                affected_rows += cursor.execute(sql, args)
            connection.commit()
            return True, affected_rows
        except Exception as exc:
            if connection is not None:
                connection.rollback()
            error = normalize_error(exc)
            logger.warning(
                "MySQL transaction failed error_type={} error_code={}",
                error["error_type"],
                error["error_code"],
            )
            return False, safe_error_status(exc)
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()

    def execute_in_transaction(self, operation: Callable[[Any], Any]) -> tuple[bool, Any]:
        connection = cursor = None
        try:
            connection = self._pool.connection()
            cursor = connection.cursor()
            result = operation(cursor)
            connection.commit()
            return True, result
        except Exception as exc:
            if connection is not None:
                connection.rollback()
            error = normalize_error(exc)
            logger.warning(
                "MySQL transaction callback failed error_type={} error_code={}",
                error["error_type"],
                error["error_code"],
            )
            return False, safe_error_status(exc)
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()


def initialize_mysql_client() -> None:
    global _mysql_client
    if _mysql_client is not None:
        return

    settings = get_settings()
    try:
        _mysql_client = MySQLClient(
            host=settings.db_host,
            port=settings.db_port,
            user=settings.db_user,
            password=settings.db_password,
            db_name=settings.db_name,
            max_connections=settings.db_pool_max_connections,
            min_cached=settings.db_pool_min_cached,
            max_cached=settings.db_pool_max_cached,
            blocking=settings.db_pool_blocking,
            connect_timeout=settings.db_connect_timeout_seconds,
            read_timeout=settings.db_read_timeout_seconds,
            write_timeout=settings.db_write_timeout_seconds,
        )
        logger.info("MySQL client initialized at {}:{}/{}", settings.db_host, settings.db_port, settings.db_name)
    except Exception as exc:
        _mysql_client = None
        error = normalize_error(exc)
        logger.warning(
            "MySQL unavailable, durable operations will return service unavailable "
            "error_type={} error_code={}",
            error["error_type"],
            error["error_code"],
        )


def get_mysql_client() -> MySQLClient | None:
    if _mysql_client is None:
        initialize_mysql_client()
    return _mysql_client


def mysql_health_status() -> dict[str, Any]:
    client = get_mysql_client()
    if client is None:
        return {"status": "unavailable"}
    ok, _ = client.execute_query("SELECT 1 AS healthy", fetch_one=True)
    return {"status": "ready" if ok else "unavailable"}
