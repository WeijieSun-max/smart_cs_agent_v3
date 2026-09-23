from __future__ import annotations

from fastapi import FastAPI

from adapter.web.routes import action_controller, chat_controller, health_controller, memory_controller, workbench_controller


def register_routes(app: FastAPI) -> None:
    app.include_router(health_controller.router, tags=["Health"])
    app.include_router(chat_controller.router, tags=["CustomerService"])
    app.include_router(action_controller.router, tags=["GovernedActions"])
    app.include_router(memory_controller.router, tags=["Memory"])
    app.include_router(workbench_controller.router, tags=["Workbench"])
