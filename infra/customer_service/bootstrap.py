from __future__ import annotations

from infra.customer_service.file_skill_bootstrap import initialize_file_skills
from infra.business import MySQLBusinessStore
from domain.business.service import initialize_service as initialize_business_service
from domain.action_governance import GovernedActionService, initialize_action_service
from domain.customer_service_agent.tools.tool_registry import get_mcp_server
from infra.db import mysql_client
from pkg.log.logger import get_logger
from pkg.config.settings import get_settings
from infra.db.migrations import apply_migrations
from application.customer_service.session_ownership import initialize_session_ownership
from application.customer_service.turn_lease_service import initialize_turn_lease_manager

logger = get_logger()


def initialize_customer_service_dependencies() -> None:
    mysql_client.initialize_mysql_client()
    apply_migrations(mysql_client.get_mysql_client(), get_settings().root_dir / "migrations")
    initialize_session_ownership(mysql_client.get_mysql_client())
    initialize_turn_lease_manager(mysql_client.get_mysql_client())

    business = initialize_business_service(MySQLBusinessStore(mysql_client.get_mysql_client()))
    initialize_action_service(
        GovernedActionService(
            get_mcp_server(),
            business,
            ttl_seconds=get_settings().pending_action_ttl_seconds,
            tool_timeout_seconds=get_settings().skill_execution_timeout_seconds,
        )
    )
    initialize_file_skills()
    logger.info("Customer service module initialized")
