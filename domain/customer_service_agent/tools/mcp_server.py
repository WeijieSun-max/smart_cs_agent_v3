from __future__ import annotations

import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from inspect import Parameter, signature
from typing import Any, Awaitable, Callable, Literal

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema.validators import validator_for

from pkg.telemetry import normalize_error, safe_error_status
from pkg.config.settings import get_settings
from pkg.exceptions.exception import ToolValidationError
from pkg.telemetry.prometheus_metrics import tool_calls_total,tool_duration_seconds

ToolEffect = Literal["read", "write"]
ConfirmationPolicy = Literal["never", "always"]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    category: str = "general"
    effect: ToolEffect = "read"
    supports_idempotency: bool = False
    version: str = "1.0.0"
    domain: str = "shared"
    capabilities: tuple[str, ...] = ()
    allowed_agent_types: tuple[str, ...] = ()
    risk_level: Literal["low", "medium", "high"] = "low"
    confirmation_policy: ConfirmationPolicy = "never"
    parallel_safe: bool = True


@dataclass
class ToolCallResult:
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_type: str | None = None
    error_code: str | None = None


class MCPToolServer:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._call_log: deque[ToolCallResult] = deque(maxlen=get_settings().tool_call_log_limit)

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        category: str = "general",
        *,
        effect: ToolEffect,
        supports_idempotency: bool,
        version: str = "1.0.0",
        domain: str = "shared",
        capabilities: tuple[str, ...] = (),
        allowed_agent_types: tuple[str, ...] = (),
        risk_level: Literal["low", "medium", "high"] = "low",
        confirmation_policy: ConfirmationPolicy | None = None,
        parallel_safe: bool | None = None,
    ) -> Callable:
        if effect not in ("read", "write"):
            raise ValueError("effect must be 'read' or 'write'")
        if type(supports_idempotency) is not bool:
            raise ValueError("supports_idempotency must be a boolean")
        resolved_confirmation = confirmation_policy or ("always" if effect == "write" else "never")
        if effect == "write" and resolved_confirmation != "always":
            raise ValueError("all write tools require confirmation")
        if effect == "write" and not supports_idempotency:
            raise ValueError("write tools must support idempotency")
        resolved_parallel_safe = effect == "read" if parallel_safe is None else parallel_safe
        if effect == "write" and resolved_parallel_safe:
            raise ValueError("write tools cannot be parallel safe")

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            if name in self._tools:
                raise ValueError(f"duplicate tool name: {name}")
            self._validate_handler_contract(func)
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                input_schema=deepcopy(input_schema),
                handler=func,
                category=category,
                effect=effect,
                supports_idempotency=supports_idempotency,
                version=version,
                domain=domain,
                capabilities=tuple(capabilities),
                allowed_agent_types=tuple(allowed_agent_types),
                risk_level=risk_level,
                confirmation_policy=resolved_confirmation,
                parallel_safe=resolved_parallel_safe,
            )
            return func

        return decorator

    def list_tools(self, category: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": deepcopy(tool.input_schema),
                "category": tool.category,
                "effect": tool.effect,
                "supportsIdempotency": tool.supports_idempotency,
                "version": tool.version,
                "domain": tool.domain,
                "capabilities": list(tool.capabilities),
                "allowedAgentTypes": list(tool.allowed_agent_types),
                "riskLevel": tool.risk_level,
                "confirmationPolicy": tool.confirmation_policy,
                "parallelSafe": tool.parallel_safe,
            }
            for tool in self._tools.values()
            if category is None or tool.category == category
        ]

    def get_tool(self, name: str) -> ToolDefinition | None:
        tool = self._tools.get(name)
        if tool is None:
            return None
        return ToolDefinition(
            name=tool.name,
            description=tool.description,
            input_schema=deepcopy(tool.input_schema),
            handler=tool.handler,
            category=tool.category,
            effect=tool.effect,
            supports_idempotency=tool.supports_idempotency,
            version=tool.version,
            domain=tool.domain,
            capabilities=tool.capabilities,
            allowed_agent_types=tool.allowed_agent_types,
            risk_level=tool.risk_level,
            confirmation_policy=tool.confirmation_policy,
            parallel_safe=tool.parallel_safe,
        )

    def validate_arguments(self, name: str, arguments: dict[str, Any]) -> None:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolValidationError()
        self._validate_arguments(tool, arguments)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        trusted_context: dict[str, Any] | None = None,
    ) -> ToolCallResult:
        tool = self._tools.get(name)
        if tool is None:
            result = ToolCallResult(
                name,
                False,
                error="ToolNotFound",
                error_type="ToolNotFoundError",
                error_code="tool.not_found",
            )
            self._call_log.append(result)
            return result

        started = time.perf_counter()
        try:
            if tool.effect == "write" and not bool((trusted_context or {}).get("governed_action")):
                raise ToolValidationError()
            self.validate_arguments(name, arguments)
            handler_arguments = dict(arguments)
            handler_arguments["_trusted_context"] = (
                None if trusted_context is None else dict(trusted_context)
            )
            output = await tool.handler(**handler_arguments)
            result = ToolCallResult(
                name,
                True,
                result=output,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            error = normalize_error(exc)
            result = ToolCallResult(
                name,
                False,
                error=safe_error_status(exc),
                duration_ms=(time.perf_counter() - started) * 1000,
                error_type=error["error_type"],
                error_code=error["error_code"],
            )
        self._call_log.append(result)
        tool_calls_total.labels(tool.name,tool.effect,"success" if result.success else "failed").inc()
        tool_duration_seconds.labels(tool.name,tool.effect).observe(result.duration_ms/1000)
        return result

    @staticmethod
    def _validate_handler_contract(func: Callable[..., Awaitable[Any]]) -> None:
        try:
            parameters = signature(func).parameters
        except (TypeError, ValueError) as exc:
            raise TypeError("tool handler signature must be inspectable") from exc
        trusted_parameter = parameters.get("_trusted_context")
        accepts_trusted_keyword = trusted_parameter is not None and trusted_parameter.kind in {
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        }
        accepts_arbitrary_keywords = any(
            parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        if not accepts_trusted_keyword and not accepts_arbitrary_keywords:
            raise TypeError("tool handler must accept _trusted_context or **kwargs")

    @staticmethod
    def _validate_arguments(tool: ToolDefinition, arguments: dict[str, Any]) -> None:
        if not isinstance(arguments, dict):
            raise ToolValidationError()
        schema = {**tool.input_schema}
        schema.setdefault("additionalProperties", False)
        validator_class = validator_for(schema)
        try:
            validator_class.check_schema(schema)
            validator_class(schema).validate(arguments)
        except JsonSchemaValidationError as exc:
            raise ToolValidationError() from exc

    def get_call_log(self, last_n: int = 100) -> list[dict[str, Any]]:
        return [
            {
                "tool": item.tool_name,
                "success": item.success,
                "duration_ms": item.duration_ms,
                "timestamp": item.timestamp,
                "error": item.error,
                "error_type": item.error_type,
                "error_code": item.error_code,
            }
            for item in list(self._call_log)[-last_n:]
        ]
