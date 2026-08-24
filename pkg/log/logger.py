from __future__ import annotations

from loguru import logger

from pkg.config.settings import get_settings

_configured = False


def setup_logger() -> None:
    global _configured
    if _configured:
        return

    settings = get_settings()
    logs_dir = settings.root_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    level = "DEBUG" if settings.debug else "INFO"

    logger.remove()
    logger.add(
        sink=lambda message: print(message, end=""),
        level=level,
        enqueue=True,
    )
    logger.add(
        logs_dir / "app.log",
        level=level,
        rotation="20 MB",
        retention="7 days",
        encoding="utf-8",
        enqueue=True,
    )
    _configured = True


def get_logger():
    return logger
