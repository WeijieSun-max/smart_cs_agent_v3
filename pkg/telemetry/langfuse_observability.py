from __future__ import annotations

import hashlib
import hmac
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Iterator, Literal, ParamSpec, TypeVar, cast
from uuid import NAMESPACE_URL, uuid4, uuid5

from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler
from langfuse.types import MaskOtelSpansParams, MaskOtelSpansResult, OtelSpanPatch
from pkg.config.settings import Settings
from pkg.log.logger import get_logger

logger = get_logger()

P = ParamSpec("P")
R = TypeVar("R")
ObservationType = Literal["tool", "retriever"]

_client: Langfuse | None = None
_enabled = False
_capture_content = False
_hash_key = b"smart-cs-agent"

_CONTENT_ATTRIBUTES = {
    "langfuse.trace.input",
    "langfuse.trace.output",
    "langfuse.observation.input",
    "langfuse.observation.output",
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "llm.prompts",
    "llm.completions",
}
_CONTENT_PREFIXES = (
    "gen_ai.prompt.",
    "gen_ai.completion.",
)
_PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_ID_CARD_PATTERN = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_BANK_CARD_PATTERN = re.compile(r"(?<!\d)\d{16,19}(?!\d)")
_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_ERROR_CODE_PATTERN = re.compile(
    r"\b(?:AllocationQuota|InvalidParameter|RateLimit|Permission|Authentication|Model|Request)"
    r"(?:\.[A-Za-z0-9_]+)+\b"
)


@dataclass(frozen=True)
class TurnTrace:
    turn_id: str
    trace_id: str
    session_id: str
    user_id: str


@dataclass
class TurnDiagnostics:
    json_parse_results: dict[str, bool] = field(default_factory=dict)
    raw_route_valid: bool | None = None
    route_fallback_used: bool = False
    tool_calls: int = 0
    tool_failures: int = 0
    retrieval_calls: int = 0
    retrieved_documents: int = 0
    fallbacks: set[str] = field(default_factory=set)
    error: dict[str, Any] | None = None

    def as_metadata(self) -> dict[str, Any]:
        return {
            "json_parse_count": len(self.json_parse_results),
            "json_parse_failure_count": sum(not value for value in self.json_parse_results.values()),
            "raw_route_valid": self.raw_route_valid,
            "route_fallback_used": self.route_fallback_used,
            "tool_call_count": self.tool_calls,
            "tool_failure_count": self.tool_failures,
            "retrieval_call_count": self.retrieval_calls,
            "retrieved_document_count": self.retrieved_documents,
            "fallback_components": sorted(self.fallbacks),
            "error": self.error,
        }


_current_diagnostics: ContextVar[TurnDiagnostics | None] = ContextVar(
    "smart_cs_turn_diagnostics",
    default=None,
)


class SafeLangfuseCallbackHandler(CallbackHandler):
    """LangChain callback that never exports the provider's raw exception text."""

    def _get_error_level_and_status_message(self, error: BaseException) -> tuple[Literal["DEFAULT", "ERROR"], str]:
        if type(error).__name__ in {"GraphBubbleUp", "GraphInterrupt", "NodeInterrupt"}:
            return "DEFAULT", type(error).__name__
        return "ERROR", safe_error_status(error)


def initialize_langfuse(settings: Settings) -> None:
    global _client, _enabled, _capture_content, _hash_key

    if _client is not None or _enabled:
        return

    _capture_content = settings.langfuse_capture_content
    hash_secret = settings.langfuse_hash_salt or settings.langfuse_secret_key
    if hash_secret:
        _hash_key = hash_secret.encode("utf-8")

    if not settings.langfuse_enabled:
        logger.info("Langfuse disabled by configuration")
        return

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning("Langfuse enabled but public/secret key is missing")
        return

    _client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        environment=settings.langfuse_environment,
        release=settings.langfuse_release,
        sample_rate=settings.langfuse_sample_rate,
        mask_otel_spans=mask_otel_spans,
    )
    _enabled = True
    logger.info(
        "Langfuse initialized environment={} release={} capture_content={}",
        settings.langfuse_environment,
        settings.langfuse_release,
        settings.langfuse_capture_content,
    )


def shutdown_langfuse() -> None:
    if _client is not None:
        _client.shutdown()


def is_langfuse_enabled() -> bool:
    return _enabled and _client is not None


def get_callback_handler() -> CallbackHandler | None:
    if not is_langfuse_enabled():
        return None
    return SafeLangfuseCallbackHandler()


def create_turn_trace(user_id: str | None, session_id: str, turn_id: str | None = None) -> TurnTrace:
    turn_id = turn_id or uuid4().hex
    return TurnTrace(
        turn_id=turn_id,
        trace_id=Langfuse.create_trace_id(seed=turn_id),
        session_id=_pseudonymize(session_id),
        user_id=_pseudonymize(user_id or "anonymous"),
    )


@contextmanager
def trace_turn(
    turn: TurnTrace,
    *,
    model: str,
    provider: str,
    input_text: str | None = None,
) -> Iterator[Any | None]:
    diagnostics = TurnDiagnostics()
    token = _current_diagnostics.set(diagnostics)
    try:
        if not is_langfuse_enabled() or _client is None:
            yield None
            return

        with _client.start_as_current_observation(
            as_type="agent",
            name="customer-service-turn",
            trace_context={"trace_id": turn.trace_id},
            input={"message": redact_sensitive_text(input_text)} if _capture_content and input_text else None,
            metadata={
                "turn_id": turn.turn_id,
                "model": model,
                "provider": provider,
            },
        ) as root:
            with propagate_attributes(
                trace_name="customer-service-turn",
                session_id=turn.session_id,
                user_id=turn.user_id,
                tags=["customer-service", "langgraph"],
                metadata={"turn_id": turn.turn_id, "provider": provider},
            ):
                yield root
    finally:
        _current_diagnostics.reset(token)


def finalize_turn(
    root: Any | None,
    *,
    workflow_completed: bool,
    response_delivered: bool,
    final_response: str = "",
    compliance_passed: bool = True,
    error: BaseException | None = None,
) -> None:
    diagnostics = _current_diagnostics.get() or TurnDiagnostics()
    if error is not None:
        diagnostics.error = normalize_error(error)

    if root is None:
        return

    update: dict[str, Any] = {"metadata": diagnostics.as_metadata()}
    if _capture_content:
        update["output"] = {"response": redact_sensitive_text(final_response)}
    if error is not None:
        update.update(level="ERROR", status_message=safe_error_status(error))
    root.update(**update)

    _score(root, "workflow_completed", workflow_completed)
    _score(root, "response_delivered", response_delivered)
    _score(root, "response_not_empty", bool(final_response.strip()))
    _score(root, "guardrail_blocked", not compliance_passed)
    _score(root, "fallback_used", bool(diagnostics.fallbacks))

    if diagnostics.raw_route_valid is not None:
        _score(root, "raw_route_valid", diagnostics.raw_route_valid)
        _score(root, "route_fallback_used", diagnostics.route_fallback_used)
    if diagnostics.json_parse_results:
        _score(root, "json_parse_all_success", all(diagnostics.json_parse_results.values()))
    if diagnostics.tool_calls:
        _score(root, "tool_success", diagnostics.tool_failures == 0)
    if diagnostics.retrieval_calls:
        _score(root, "rag_has_context", diagnostics.retrieved_documents > 0)


def submit_user_feedback(trace_id: str, helpful: bool, reason: str | None = None) -> bool:
    if not is_langfuse_enabled() or _client is None:
        return False
    score_id = uuid5(NAMESPACE_URL, f"smart-cs-feedback:{trace_id}").hex
    _client.create_score(
        trace_id=trace_id,
        score_id=score_id,
        name="user_feedback",
        value=1.0 if helpful else 0.0,
        data_type="BOOLEAN",
        comment=reason,
        metadata={"reason": reason} if reason else None,
    )
    return True


def record_json_parse(stage: str, success: bool) -> None:
    diagnostics = _current_diagnostics.get()
    if diagnostics is not None:
        diagnostics.json_parse_results[stage] = success


def record_route(valid: bool, fallback_used: bool) -> None:
    diagnostics = _current_diagnostics.get()
    if diagnostics is not None:
        diagnostics.raw_route_valid = valid
        diagnostics.route_fallback_used = fallback_used


def record_fallback(component: str) -> None:
    diagnostics = _current_diagnostics.get()
    if diagnostics is not None:
        diagnostics.fallbacks.add(component)


def traced_dependency(name: str, observation_type: ObservationType) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Trace a synchronous service call without exporting arguments or result bodies."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            diagnostics = _current_diagnostics.get()
            if observation_type == "tool" and diagnostics is not None:
                diagnostics.tool_calls += 1
            if observation_type == "retriever" and diagnostics is not None:
                diagnostics.retrieval_calls += 1

            if not is_langfuse_enabled() or _client is None:
                try:
                    result = func(*args, **kwargs)
                except Exception:
                    if observation_type == "tool" and diagnostics is not None:
                        diagnostics.tool_failures += 1
                    raise
                _record_dependency_result(observation_type, diagnostics, result)
                return result

            caught: tuple[Exception, Any] | None = None
            result: R | None = None
            with _client.start_as_current_observation(
                as_type=observation_type,
                name=name,
                metadata={"argument_count": len(args) + len(kwargs)},
            ) as observation:
                try:
                    result = func(*args, **kwargs)
                except Exception as exc:
                    caught = (exc, exc.__traceback__)
                    observation.update(
                        level="ERROR",
                        status_message=safe_error_status(exc),
                        metadata={"error": normalize_error(exc)},
                    )
                    if observation_type == "tool" and diagnostics is not None:
                        diagnostics.tool_failures += 1
                else:
                    metadata = _dependency_metadata(observation_type, result)
                    observation.update(metadata=metadata)
                    _record_dependency_result(observation_type, diagnostics, result)

            if caught is not None:
                exc, traceback = caught
                raise exc.with_traceback(traceback)
            return cast(R, result)

        return wrapper

    return decorator


def normalize_error(error: BaseException) -> dict[str, Any]:
    code = getattr(error, "code", None)
    body = getattr(error, "body", None)
    if not code and isinstance(body, dict):
        code = body.get("code")
        nested_error = body.get("error")
        if not code and isinstance(nested_error, dict):
            code = nested_error.get("code")
    if not code:
        match = _ERROR_CODE_PATTERN.search(str(error))
        code = match.group(0) if match else None

    response = getattr(error, "response", None)
    status_code = getattr(error, "status_code", None) or getattr(response, "status_code", None)
    return {
        "error_type": type(error).__name__,
        "error_code": str(code)[:120] if code else None,
        "http_status_code": int(status_code) if isinstance(status_code, int) else None,
    }


def safe_error_status(error: BaseException) -> str:
    normalized = normalize_error(error)
    code = normalized.get("error_code")
    return f"{normalized['error_type']}:{code}" if code else str(normalized["error_type"])


def redact_sensitive_text(value: str) -> str:
    redacted = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", value)
    redacted = _ID_CARD_PATTERN.sub("[REDACTED_ID]", redacted)
    redacted = _BANK_CARD_PATTERN.sub("[REDACTED_BANK_CARD]", redacted)
    return _PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)


def mask_otel_spans(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult | None:
    patches: dict[Any, OtelSpanPatch] = {}
    for identifier, span in params.spans.items():
        delete_attributes: list[str] = []
        set_attributes: dict[str, Any] = {}
        for key, value in span.attributes.items():
            lowered = key.lower()
            is_content = key in _CONTENT_ATTRIBUTES or lowered.startswith(_CONTENT_PREFIXES)
            if is_content and not _capture_content:
                delete_attributes.append(key)
                continue
            if isinstance(value, str):
                masked = redact_sensitive_text(value)
                if masked != value:
                    set_attributes[key] = masked
        if delete_attributes or set_attributes:
            patches[identifier] = OtelSpanPatch(
                delete_attributes=tuple(delete_attributes),
                set_attributes=set_attributes,
            )
    return MaskOtelSpansResult(span_patches=patches) if patches else None


def _pseudonymize(value: str) -> str:
    return hmac.new(_hash_key, value.encode("utf-8"), hashlib.sha256).hexdigest()


def _score(root: Any, name: str, value: bool) -> None:
    root.score_trace(name=name, value=1.0 if value else 0.0, data_type="BOOLEAN")


def _dependency_metadata(observation_type: ObservationType, result: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {"result_type": type(result).__name__, "success": True}
    if observation_type == "retriever" and isinstance(result, list):
        metadata["document_count"] = len(result)
        metadata["sources"] = sorted({str(item.get("source")) for item in result if isinstance(item, dict) and item.get("source")})[:20]
    return metadata


def _record_dependency_result(
    observation_type: ObservationType,
    diagnostics: TurnDiagnostics | None,
    result: Any,
) -> None:
    if observation_type == "retriever" and diagnostics is not None and isinstance(result, list):
        diagnostics.retrieved_documents += len(result)
