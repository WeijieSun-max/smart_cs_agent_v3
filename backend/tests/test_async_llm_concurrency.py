from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from domain.shared.llm import llm_service


class AsyncClient:
    def __init__(self) -> None:
        self.active = 0
        self.maximum = 0

    async def ainvoke(self, _messages, config=None):
        del config
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return SimpleNamespace(content="ok")


def test_async_calls_share_a_hard_nonblocking_limit(monkeypatch) -> None:
    client = AsyncClient()
    monkeypatch.setattr(llm_service, "instance", client)
    monkeypatch.setattr(llm_service, "_profiles", {"default": client})
    monkeypatch.setattr(llm_service, "_profile_by_run_prefix", {})
    monkeypatch.setattr(llm_service, "_slots", threading.BoundedSemaphore(2))
    monkeypatch.setattr(llm_service, "_queue_capacity", 10)
    monkeypatch.setattr(llm_service, "_queued_calls", 0)
    monkeypatch.setattr(llm_service, "_queue_timeout_seconds", 1.0)

    async def scenario() -> None:
        await asyncio.gather(*[
            llm_service.ainvoke_llm([], run_name=f"test.{index}")
            for index in range(8)
        ])

    asyncio.run(scenario())
    assert client.maximum == 2


def test_async_llm_queue_is_bounded(monkeypatch) -> None:
    release = asyncio.Event()

    class BlockingClient:
        async def ainvoke(self, _messages, config=None):
            del config
            await release.wait()
            return SimpleNamespace(content="ok")

    client = BlockingClient()
    monkeypatch.setattr(llm_service, "instance", client)
    monkeypatch.setattr(llm_service, "_profiles", {"default": client})
    monkeypatch.setattr(llm_service, "_profile_by_run_prefix", {})
    monkeypatch.setattr(llm_service, "_slots", threading.BoundedSemaphore(1))
    monkeypatch.setattr(llm_service, "_queue_capacity", 1)
    monkeypatch.setattr(llm_service, "_queued_calls", 0)
    monkeypatch.setattr(llm_service, "_queue_timeout_seconds", 1.0)

    async def scenario() -> None:
        active = asyncio.create_task(llm_service.ainvoke_llm([], run_name="active"))
        await asyncio.sleep(0)
        queued = asyncio.create_task(llm_service.ainvoke_llm([], run_name="queued"))
        while llm_service._queued_calls != 1:
            await asyncio.sleep(0)
        try:
            await llm_service.ainvoke_llm([], run_name="rejected")
        except RuntimeError as exc:
            assert str(exc) == "LLM admission queue is full"
        else:
            raise AssertionError("third call should be rejected")
        release.set()
        await asyncio.gather(active, queued)

    asyncio.run(scenario())
