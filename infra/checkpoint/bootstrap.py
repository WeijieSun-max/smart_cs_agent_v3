from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from domain.shared.checkpoint.checkpoint_saver_service import initialize_service
from pkg.log.logger import get_logger
from infra.db.mysql_client import get_mysql_client
from infra.checkpoint.mysql_checkpoint_saver import MySQLCheckpointSaver

logger = get_logger()


def initialize_checkpointing() -> None:
    # 创建 LangGraph 的内存 checkpoint MemorySaver，
    # 注入到共享服务里 initialize_service(MemorySaver())
    client = get_mysql_client()
    if client is not None:
        initialize_service(MySQLCheckpointSaver(client))
        logger.info("Checkpoint module initialized with durable MySQL saver")
    else:
        initialize_service(MemorySaver())
        logger.warning("MySQL unavailable; checkpoint module degraded to in-memory saver")
