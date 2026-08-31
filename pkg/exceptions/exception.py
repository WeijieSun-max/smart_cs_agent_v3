from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from pkg.log.logger import get_logger
from pkg.telemetry import normalize_error

logger = get_logger()
SAFE_ERROR_MESSAGE = "系统处理异常，请稍后重试。"


class ServiceError(Exception):
    code = "service.error"
    status_code = 500
    safe_message = SAFE_ERROR_MESSAGE


class StorageUnavailableError(ServiceError):
    code = "storage.unavailable"
    status_code = 503
    safe_message = "持久化存储暂时不可用，请稍后重试。"


class StorageOperationError(ServiceError):
    code = "storage.operation_failed"
    status_code = 503
    safe_message = "数据未能持久化，请稍后重试。"


class UnsafeInputError(ServiceError):
    code = "input.unsafe"
    status_code = 400
    safe_message = "输入包含敏感个人信息，请脱敏后重试。"


class ToolValidationError(ServiceError):
    code = "tool.validation"
    status_code = 422
    safe_message = "工具参数不符合要求。"


class RequestConflictError(ServiceError):
    code = "request.conflict"
    status_code = 409
    safe_message = "同一请求标识不能用于不同内容。"


def handle_global_exception(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def service_error_handler(_: Request, exc: ServiceError):
        error = normalize_error(exc)
        logger.error("Service error type={} code={}", error["error_type"], error["error_code"])
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.safe_message})

    @app.exception_handler(Exception)
    async def global_error_handler(_: Request, exc: Exception):
        error = normalize_error(exc)
        logger.error("Unhandled error type={} code={}", error["error_type"], error["error_code"])
        return JSONResponse(status_code=500, content={"detail": SAFE_ERROR_MESSAGE})
