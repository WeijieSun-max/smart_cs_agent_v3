from __future__ import annotations

import multiprocessing
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from adapter.web.routes import register_routes
from application.customer_service.memory_runtime import configure_memory_workers, start_memory_workers, stop_memory_workers
from domain.customer_service_agent.workflow import customer_service_workflow
from infra import initialize_infrastructure
from infra.checkpoint.bootstrap import initialize_checkpointing
from infra.llm.bootstrap import initialize_llm
from pkg.config.settings import Settings, get_settings
from pkg.exceptions.exception import handle_global_exception
from pkg.log.logger import get_logger, setup_logger
from pkg.telemetry import initialize_langfuse, shutdown_langfuse
from adapter.web.middleware.prometheus import PrometheusMiddleware
from application.customer_service.action_runtime import start_action_worker,stop_action_worker

logger = get_logger()


def configure_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_origin_regex=r"^https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d{1,5})?$",
        # This demo has no browser authentication. Restrict cross-origin
        # callers to the local machine so arbitrary websites cannot invoke
        # ticket/session mutations through a user's browser.
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def initialize_application(app: FastAPI, settings: Settings) -> None:
    setup_logger()
    app.debug = settings.debug
    app.title = settings.app_name
    initialize_langfuse(settings)
    configure_cors(app)
    app.add_middleware(PrometheusMiddleware)
    handle_global_exception(app)
    initialize_infrastructure()
    initialize_llm()
    configure_memory_workers()
    initialize_checkpointing()
    customer_service_workflow.initialize_workflow()
    register_routes(app)
    logger.info("Application initialized")


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        start_memory_workers()
        start_action_worker()
        try:
            yield
        finally:
            stop_memory_workers()
            stop_action_worker()
            shutdown_langfuse()

    app = FastAPI(lifespan=lifespan)
    initialize_application(app, get_settings())
    return app


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    multiprocessing.freeze_support()
    settings = get_settings()
    uvicorn.run(
        app="main:create_app",
        host=settings.server_host,
        port=settings.server_port,
        reload=False,
        factory=True,
    )
