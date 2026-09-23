from __future__ import annotations

from infra.customer_service.bootstrap import initialize_customer_service_dependencies
from infra.knowledge.bootstrap import initialize_knowledge_store
from infra.memory.bootstrap import initialize_memory
from pkg.log.logger import get_logger

logger = get_logger()


def initialize_infrastructure() -> None:
    initialize_memory()
    initialize_knowledge_store()
    initialize_customer_service_dependencies()
    logger.info("Infrastructure modules initialized")
