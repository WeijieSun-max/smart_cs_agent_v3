"""管理 LLM profile 路由、全局并发槽和稳定可观测元数据。"""

from __future__ import annotations

import asyncio
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
_queue_capacity = 40
_queued_calls = 0


@dataclass(frozen=True)
class ModelProfile:
    """一个 OpenAI-compatible 模型端点的不可变运行参数。"""

    name: str
    model: str
    base_url: str
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 1
    enable_thinking: bool = False


def initialize_llm_client(llm_client: BaseChatModel) -> None:
    """安装单一默认模型，主要供确定性测试和简单调用方使用。"""

    global instance
    instance = llm_client
    _profiles["default"] = llm_client


def initialize_llm_profiles(
    clients: Mapping[str, BaseChatModel],
    *,
    run_prefixes: Mapping[str, str],
    max_concurrency: int = 20,
    queue_capacity: int = 40,
    queue_timeout_seconds: float = 15.0,
) -> None:
    """安装运行期间不可变的节点模型 profile 注册表。

    run_name 使用最长前缀匹配；所有映射必须指向已注册 profile。全局有界信号
    量限制并发模型调用，队列等待超过配置时间会明确失败而不是无限堆积。
    """
    if "default" not in clients:
        raise ValueError("model profile registry requires a default client")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    if queue_capacity < 0:
        raise ValueError("queue_capacity cannot be negative")
    global instance, _profiles, _profile_by_run_prefix, _slots, _slot_limit
    global _queue_timeout_seconds, _queue_capacity, _queued_calls
    instance = clients["default"]
    _profiles = dict(clients)
    unknown = set(run_prefixes.values()) - set(_profiles)
    if unknown:
        raise ValueError(f"unknown model profiles: {sorted(unknown)}")
    _profile_by_run_prefix = dict(run_prefixes)
    _slots = threading.BoundedSemaphore(max_concurrency)
    _slot_limit = max_concurrency
    _queue_capacity = queue_capacity
    _queued_calls = 0
    _queue_timeout_seconds = float(queue_timeout_seconds)


def get_llm_client(profile: str | None = None, run_name: str | None = None) -> BaseChatModel:
    """按显式 profile 或 run_name 前缀选择模型，最后回退到 default。"""

    if instance is None:
        raise RuntimeError("LLM client is not initialized")
    selected = profile or _profile_for_run(run_name or "")
    return _profiles.get(selected, instance)


def _profile_for_run(run_name: str) -> str:
    """以最长匹配前缀解析节点 profile，避免短前缀遮蔽专用节点。"""

    matches = [prefix for prefix in _profile_by_run_prefix if run_name.startswith(prefix)]
    return _profile_by_run_prefix[max(matches, key=len)] if matches else "default"


def _acquire_slot() -> threading.BoundedSemaphore | None:
    """在超时内取得全局 LLM 并发槽，并记录排队/在途指标。"""

    semaphore = _slots
    if semaphore is not None:
        if semaphore.acquire(blocking=False):
            llm_inflight.inc()
            return semaphore
        _reserve_queue_slot()
        started = time.perf_counter()
        llm_queue_depth.inc()
        try:
            if not semaphore.acquire(timeout=_queue_timeout_seconds):
                raise RuntimeError("LLM admission queue timeout")
        finally:
            _release_queue_slot()
            llm_queue_depth.dec()
            llm_queue_wait_seconds.observe(time.perf_counter() - started)
        llm_inflight.inc()
    return semaphore


async def _acquire_slot_async() -> threading.BoundedSemaphore | None:
    """Acquire the shared thread-safe limiter without blocking the event loop."""

    semaphore = _slots
    if semaphore is None:
        return None
    if semaphore.acquire(blocking=False):
        llm_inflight.inc()
        return semaphore
    _reserve_queue_slot()
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + _queue_timeout_seconds
    llm_queue_depth.inc()
    try:
        while not semaphore.acquire(blocking=False):
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise RuntimeError("LLM admission queue timeout")
            await asyncio.sleep(min(0.01, remaining))
    finally:
        _release_queue_slot()
        llm_queue_depth.dec()
        llm_queue_wait_seconds.observe(loop.time() - started)
    llm_inflight.inc()
    return semaphore


def _reserve_queue_slot() -> None:
    global _queued_calls
    with _slot_lock:
        if _queued_calls >= _queue_capacity:
            raise RuntimeError("LLM admission queue is full")
        _queued_calls += 1


def _release_queue_slot() -> None:
    global _queued_calls
    with _slot_lock:
        _queued_calls -= 1


def invoke_llm(
    messages: Sequence[BaseMessage],
    *,
    run_name: str,
    prompt_version: str = "v1",
) -> Any:
    """调用选定模型并附加稳定的提示版本、节点和 profile 元数据。"""
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


async def ainvoke_llm(
    messages: Sequence[BaseMessage],
    *,
    run_name: str,
    prompt_version: str = "v1",
) -> Any:
    """Asynchronously invoke a model under the same global limiter as sync jobs."""

    semaphore = await _acquire_slot_async()
    try:
        return await get_llm_client(run_name=run_name).ainvoke(
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
