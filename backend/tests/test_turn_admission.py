from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from application.customer_service import admission


def _settings(**overrides):
    values = {
        "turn_max_concurrency": 2,
        "turn_queue_capacity": 4,
        "turn_queue_timeout_seconds": 1.0,
        "per_session_queue_limit": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_global_active_turns_are_hard_bounded(monkeypatch) -> None:
    monkeypatch.setattr(admission, "get_settings", lambda: _settings())
    controller = admission.TurnAdmissionController()
    active = 0
    maximum = 0
    release = asyncio.Event()

    async def worker(index: int) -> None:
        nonlocal active, maximum
        async with controller.slot("user-1", f"session-{index}"):
            active += 1
            maximum = max(maximum, active)
            await release.wait()
            active -= 1

    async def scenario() -> None:
        tasks = [asyncio.create_task(worker(index)) for index in range(6)]
        while controller.snapshot() != {"active": 2, "queued": 4, "active_sessions": 2}:
            await asyncio.sleep(0)
        release.set()
        await asyncio.gather(*tasks)

    asyncio.run(scenario())
    assert maximum == 2


def test_same_session_is_serial_but_other_sessions_can_run(monkeypatch) -> None:
    monkeypatch.setattr(admission, "get_settings", lambda: _settings(turn_max_concurrency=3))
    controller = admission.TurnAdmissionController()
    session_active = 0
    session_maximum = 0
    total_active = 0
    total_maximum = 0

    async def worker(session_id: str) -> None:
        nonlocal session_active, session_maximum, total_active, total_maximum
        async with controller.slot("user-1", session_id):
            total_active += 1
            total_maximum = max(total_maximum, total_active)
            if session_id == "shared":
                session_active += 1
                session_maximum = max(session_maximum, session_active)
            await asyncio.sleep(0.01)
            if session_id == "shared":
                session_active -= 1
            total_active -= 1

    async def scenario() -> None:
        await asyncio.gather(worker("shared"), worker("shared"), worker("other"))

    asyncio.run(scenario())
    assert session_maximum == 1
    assert total_maximum == 2


def test_full_queue_fails_fast(monkeypatch) -> None:
    monkeypatch.setattr(
        admission,
        "get_settings",
        lambda: _settings(turn_max_concurrency=1, turn_queue_capacity=1),
    )
    controller = admission.TurnAdmissionController()
    release = asyncio.Event()

    async def holder() -> None:
        async with controller.slot("user-1", "active"):
            await release.wait()

    async def waiter() -> None:
        async with controller.slot("user-1", "queued"):
            pass

    async def scenario() -> None:
        holding = asyncio.create_task(holder())
        while controller.snapshot()["active"] != 1:
            await asyncio.sleep(0)
        waiting = asyncio.create_task(waiter())
        while controller.snapshot()["queued"] != 1:
            await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="agent admission queue is full"):
            async with controller.slot("user-1", "rejected"):
                pass
        release.set()
        await asyncio.gather(holding, waiting)

    asyncio.run(scenario())
