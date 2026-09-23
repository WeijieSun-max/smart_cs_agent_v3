from __future__ import annotations

from application.customer_service.memory_extraction_worker import MemoryExtractionWorker
from application.customer_service.memory_index_worker import MemoryIndexWorker
from application.customer_service.summary_update_worker import SummaryUpdateWorker
from domain.customer_service_agent.service import (
    memory_extraction_service,
    memory_service,
    session_summary_service,
    short_term_memory_service,
)
from pkg.config.settings import get_settings

_workers: list[object] = []


def configure_memory_workers() -> None:
    global _workers
    stop_memory_workers()
    _workers = []
    settings = get_settings()
    service = memory_service.get_service_or_none()
    if not settings.memory_layered_enabled or not settings.memory_worker_enabled or service is None:
        return
    _workers.append(MemoryExtractionWorker(
        service.repository,
        memory_extraction_service.MemoryExtractionService.default(),
        lease_seconds=settings.memory_worker_lease_seconds,
        max_attempts=settings.memory_worker_max_attempts,
        poll_seconds=settings.memory_worker_poll_seconds,
        extraction_increment_turns=settings.memory_extraction_increment_turns,
    ))
    _workers.append(SummaryUpdateWorker(
        service.repository,
        session_summary_service.SessionSummaryService(),
        lease_seconds=settings.memory_worker_lease_seconds,
        max_attempts=settings.memory_worker_max_attempts,
        poll_seconds=settings.memory_worker_poll_seconds,
        summary_cache=short_term_memory_service.get_service(),
        summary_eligible_turns=settings.memory_summary_eligible_turns,
        summary_increment_turns=settings.memory_summary_increment_turns,
    ))
    if service.vector_index is not None and service.vector_index.available and service.embedder is not None:
        _workers.append(MemoryIndexWorker(
            service.repository,
            service.vector_index,
            service.embedder,
            lease_seconds=settings.memory_worker_lease_seconds,
            max_attempts=settings.memory_worker_max_attempts,
            poll_seconds=settings.memory_worker_poll_seconds,
        ))


def start_memory_workers() -> None:
    for worker in _workers:
        worker.start()


def stop_memory_workers() -> None:
    for worker in _workers:
        worker.stop()


def worker_statuses() -> dict[str, dict[str, object]]:
    settings = get_settings()
    if not settings.memory_layered_enabled or not settings.memory_worker_enabled:
        return {
            "extraction": {"status": "disabled", "running": False},
            "summary": {"status": "disabled", "running": False},
            "index": {"status": "disabled", "running": False},
        }
    statuses = {
        "extraction": {"status": "degraded", "running": False},
        "summary": {"status": "degraded", "running": False},
        "index": {"status": "degraded", "running": False},
    }
    for worker in _workers:
        if isinstance(worker, MemoryExtractionWorker):
            key = "extraction"
        elif isinstance(worker, SummaryUpdateWorker):
            key = "summary"
        else:
            key = "index"
        statuses[key] = worker.health_status()
    if not settings.memory_qdrant_enabled:
        statuses["index"] = {"status": "disabled", "running": False}
    return statuses
