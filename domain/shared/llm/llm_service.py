from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pkg.telemetry.prometheus_metrics import llm_inflight,llm_queue_depth,llm_queue_wait_seconds

instance: Optional[BaseChatModel] = None
_profiles: dict[str, BaseChatModel] = {}
_profile_by_run_prefix: dict[str, str] = {}
_slot_lock = threading.Lock()
_slots: threading.BoundedSemaphore | None = None
_slot_limit = 0
_queue_timeout_seconds = 15.0


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model: str
    base_url: str
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 1


def initialize_llm_client(llm_client: BaseChatModel) -> None:
    global instance
    instance = llm_client
    _profiles["default"] = llm_client


def initialize_llm_profiles(
    clients: Mapping[str, BaseChatModel],
    *,
    run_prefixes: Mapping[str, str],
    max_concurrency: int = 20,
    queue_timeout_seconds: float = 15.0,
) -> None:
    """Install an immutable-at-runtime node profile registry.

    Longest run-name prefix wins. ``initialize_llm_client`` remains supported for
    deterministic unit tests and callers that only need a default model.
    """
    if "default" not in clients:
        raise ValueError("model profile registry requires a default client")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    global instance, _profiles, _profile_by_run_prefix, _slots, _slot_limit, _queue_timeout_seconds
    instance = clients["default"]
    _profiles = dict(clients)
    unknown = set(run_prefixes.values()) - set(_profiles)
    if unknown:
        raise ValueError(f"unknown model profiles: {sorted(unknown)}")
    _profile_by_run_prefix = dict(run_prefixes)
    _slots = threading.BoundedSemaphore(max_concurrency)
    _slot_limit = max_concurrency
    _queue_timeout_seconds = float(queue_timeout_seconds)


def get_llm_client(profile: str | None = None, run_name: str | None = None) -> BaseChatModel:
    if instance is None:
        raise RuntimeError("LLM client is not initialized")
    selected = profile or _profile_for_run(run_name or "")
    return _profiles.get(selected, instance)


def _profile_for_run(run_name: str) -> str:
    matches = [prefix for prefix in _profile_by_run_prefix if run_name.startswith(prefix)]
    return _profile_by_run_prefix[max(matches, key=len)] if matches else "default"


def model_profile_snapshot() -> dict[str, str]:
    return {prefix: profile for prefix, profile in sorted(_profile_by_run_prefix.items())}


def invoke_structured_llm(
    messages: Sequence[BaseMessage],
    *,
    schema: type[Any] | dict[str, Any],
    run_name: str,
    prompt_version: str = "v1",
) -> Any:
    """Use Qwen's OpenAI-compatible JSON Schema structured output path."""
    semaphore = _acquire_slot()
    try:
        structured = get_llm_client(run_name=run_name).with_structured_output(
            schema,
            method="json_schema",
        )
        return structured.invoke(
            messages,
            config={
                "run_name": run_name,
                "metadata": {
                    "prompt_name": run_name,
                    "prompt_version": prompt_version,
                    "model_profile": _profile_for_run(run_name),
                    "structured_output": "json_schema",
                },
            },
        )
    finally:
        if semaphore is not None:
            llm_inflight.dec()
            semaphore.release()


def _acquire_slot() -> threading.BoundedSemaphore | None:
    semaphore = _slots
    if semaphore is not None:
        started=time.perf_counter(); llm_queue_depth.inc()
        try:
            if not semaphore.acquire(timeout=_queue_timeout_seconds):
                raise RuntimeError("LLM admission queue timeout")
        finally:
            llm_queue_depth.dec(); llm_queue_wait_seconds.observe(time.perf_counter()-started)
        llm_inflight.inc()
    return semaphore


def invoke_llm(
    messages: Sequence[BaseMessage],
    *,
    run_name: str,
    prompt_version: str = "v1",
) -> Any:
    """Invoke the shared chat model with stable telemetry names."""
    semaphore = _acquire_slot()
    try:
        return get_llm_client(run_name=run_name).invoke(
            messages,
            config={
                "run_name": run_name,
                "metadata": {
                    "prompt_name": run_name,
                    "prompt_version": prompt_version,
                    "model_profile": _profile_for_run(run_name),
                },
            },
        )
    finally:
        if semaphore is not None:
            llm_inflight.dec()
            semaphore.release()


async def ainvoke_private_llm(
    messages: Sequence[BaseMessage],
    *,
    run_name: str,
    prompt_version: str = "v1",
) -> Any:
    """Invoke without inherited callbacks.

    This isolates parent Runnable callbacks only. Model-local or globally configured
    tracing remains the responsibility of the model/provider configuration.
    """
    semaphore = await __import__("asyncio").to_thread(_acquire_slot)
    try:
        return await get_llm_client(run_name=run_name).ainvoke(
            messages,
            config={
                "run_name": run_name,
                "callbacks": [],
                "metadata": {
                    "prompt_name": run_name,
                    "prompt_version": prompt_version,
                    "content_capture": False,
                    "model_profile": _profile_for_run(run_name),
                },
            },
        )
    finally:
        if semaphore is not None:
            llm_inflight.dec()
            semaphore.release()
