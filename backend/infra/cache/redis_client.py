from __future__ import annotations

from redis import ConnectionPool, Redis

from pkg.config.settings import get_settings
from pkg.log.logger import get_logger
from pkg.telemetry import normalize_error

logger = get_logger()
_redis_client: Redis | None = None
_redis_pool: ConnectionPool | None = None
_initialized = False


def initialize_redis_client() -> None:
    global _redis_client, _redis_pool, _initialized
    if _initialized:
        return
    _initialized = True

    settings = get_settings()
    if not settings.redis_enabled:
        logger.info("Redis disabled by configuration")
        return

    try:
        _redis_pool = ConnectionPool(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password or None,
            max_connections=settings.redis_max_connections,
            health_check_interval=settings.redis_health_check_interval,
            socket_connect_timeout=2,
            socket_timeout=2,
            encoding="utf-8",
            encoding_errors="strict",
            decode_responses=True,
        )
        _redis_client = Redis(connection_pool=_redis_pool)
        _redis_client.ping()
        logger.info("Redis client initialized at {}:{}", settings.redis_host, settings.redis_port)
    except Exception as exc:
        _redis_client = None
        if _redis_pool is not None:
            _redis_pool.disconnect()
        _redis_pool = None
        error = normalize_error(exc)
        logger.warning(
            "Redis unavailable, falling back to in-memory services "
            "error_type={} error_code={}",
            error["error_type"],
            error["error_code"],
        )


def get_redis_client() -> Redis | None:
    if not _initialized:
        initialize_redis_client()
    return _redis_client


def redis_health_status() -> dict[str, str]:
    client = get_redis_client()
    if client is None:
        return {"status": "disabled" if not get_settings().redis_enabled else "degraded"}
    try:
        client.ping()
    except Exception:
        return {"status": "degraded"}
    return {"status": "ready"}
