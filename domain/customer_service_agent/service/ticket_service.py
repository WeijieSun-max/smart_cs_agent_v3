from __future__ import annotations

from typing import Any, Optional

from domain.customer_service_agent.interfaces.i_ticket_repository import ITicketRepository
from pkg.telemetry import traced_dependency


class TicketService:
    def __init__(self, repository: ITicketRepository):
        self.repository = repository

    @traced_dependency("ticket.create", "tool")
    def create(
        self,
        ticket_type: str,
        priority: str,
        summary: str,
        details: str,
        user_id: str,
        operation_key: str | None = None,
    ) -> dict[str, Any]:
        return self.repository.create(ticket_type, priority, summary, details, user_id, operation_key)

    @traced_dependency("ticket.query", "tool")
    def query(self, ticket_id: str) -> Optional[dict[str, Any]]:
        return self.repository.query(ticket_id)

    @traced_dependency("ticket.update", "tool")
    def update_status(self, ticket_id: str, status: str) -> Optional[dict[str, Any]]:
        return self.repository.update_status(ticket_id, status)


instance: Optional[TicketService] = None


def initialize_service(repository: ITicketRepository) -> None:
    global instance
    instance = TicketService(repository)


def get_service() -> TicketService:
    if instance is None:
        raise RuntimeError("Ticket service is not initialized")
    return instance
