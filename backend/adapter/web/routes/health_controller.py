from __future__ import annotations

from fastapi import APIRouter, Response, status

from application.customer_service import memory_runtime
from domain.customer_service_agent.service import memory_service
from domain.customer_service_agent.service import knowledge_service
from domain.customer_service_agent.workflow import customer_service_workflow
from domain.shared.llm import llm_service
from infra.cache.redis_client import redis_health_status
from infra.customer_service.file_skill_bootstrap import file_skill_health_status
from infra.db.mysql_client import mysql_health_status
from pkg.config.settings import get_settings
from pkg.telemetry.prometheus_metrics import metrics_payload

router = APIRouter()


@router.get("/metrics",include_in_schema=False)
def prometheus_metrics():
    payload,content_type=metrics_payload()
    return Response(content=payload,media_type=content_type)


@router.get("/health/live")
def liveness():
    return {"status": "healthy", "version": "1.0.0"}


@router.get("/health/ready")
def readiness(response: Response):
    components = _component_statuses()
    ready = (
        components["mysql"]["status"] == "ready"
        and components["workflow"]["status"] == "ready"
        and components["llm"]["status"] == "ready"
        and components["skills"]["status"] in {"ready","disabled"}
    )
    if get_settings().memory_layered_enabled:
        ready = ready and components["memory_mysql"]["status"] == "ready"
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    healthy_states = {"ready", "disabled"}
    overall = "ready" if ready and all(item["status"] in healthy_states for item in components.values()) else "degraded"
    if not ready:
        overall = "unavailable"
    return {"status": overall, "ready": ready, "version": "1.0.0", "components": components}


@router.get("/health")
def health_check(response: Response):
    return readiness(response)


def _component_statuses() -> dict[str, dict[str, object]]:
    settings = get_settings()
    try:
        embedding = knowledge_service.get_service().health_status()
    except RuntimeError:
        embedding = {"status": "unavailable"}
    try:
        customer_service_workflow.get_workflow()
        workflow = {"status": "ready"}
    except RuntimeError:
        workflow = {"status": "unavailable"}
    service = memory_service.get_service_or_none()
    if not settings.memory_layered_enabled:
        memory_mysql = {"status": "disabled"}
        memory_qdrant = {"status": "disabled"}
    else:
        memory_mysql = {
            "status": "ready"
            if service is not None and service.repository.available
            else "unavailable"
        }
        if not settings.memory_qdrant_enabled:
            memory_qdrant = {"status": "disabled"}
        elif service is None or service.vector_index is None:
            memory_qdrant = {"status": "degraded"}
        else:
            try:
                memory_qdrant = service.vector_index.health_status()
                if memory_qdrant.get("status") != "ready":
                    memory_qdrant = {**memory_qdrant, "status": "degraded"}
            except Exception:
                memory_qdrant = {"status": "degraded"}
    workers = memory_runtime.worker_statuses()
    return {
        "mysql": mysql_health_status(),
        "redis": redis_health_status(),
        "llm": {"status": "ready" if llm_service.instance is not None else "unavailable"},
        "embedding": embedding,
        "workflow": workflow,
        "memory_mysql": memory_mysql,
        "memory_qdrant": memory_qdrant,
        "memory_extraction_worker": workers["extraction"],
        "memory_index_worker": workers["index"],
        "skills": file_skill_health_status(),
    }
