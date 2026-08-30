from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from infra.memory.short_term_memory import RedisShortTermMemory


def test_fallback_memory_keeps_concurrent_users_isolated() -> None:
    memory = RedisShortTermMemory(None, max_turns=200)

    def append(user_id: str, index: int) -> None:
        memory.add_message(
            "shared-session-name",
            "user",
            f"{user_id}-{index}",
            f"turn-{user_id}-{index}",
            user_id=user_id,
        )

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(append, user_id, index)
            for user_id in ("user-a", "user-b")
            for index in range(50)
        ]
        for future in futures:
            future.result()

    first = memory.get_history("shared-session-name", user_id="user-a")
    second = memory.get_history("shared-session-name", user_id="user-b")
    assert len(first) == len(second) == 50
    assert all(message["content"].startswith("user-a-") for message in first)
    assert all(message["content"].startswith("user-b-") for message in second)
    assert memory.get_session("shared-session-name", user_id="user-a")["message_count"] == 50
    assert memory.get_session("shared-session-name", user_id="user-b")["message_count"] == 50
