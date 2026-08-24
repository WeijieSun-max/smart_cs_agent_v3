from types import SimpleNamespace

import pytest

from domain.customer_service_agent.service import session_summary_service


MESSAGES = [
    {"message_id": 1, "role": "user", "content": "我想开户", "turn_id": "turn-1"},
    {"message_id": 2, "role": "assistant", "content": "请先完成风险测评", "turn_id": "turn-1"},
]


def test_summary_service_builds_incremental_structured_summary(monkeypatch) -> None:
    monkeypatch.setattr(
        session_summary_service,
        "invoke_llm",
        lambda messages, run_name, prompt_version="v1": SimpleNamespace(content='''{
            "summary_text":"用户准备开户，尚未完成风险测评。",
            "goals":["开户"],
            "entities":{},
            "actions":["提供风险测评指引"],
            "results":[],
            "missing_info":[],
            "pending_tasks":["完成风险测评"]
        }'''),
    )

    summary = session_summary_service.SessionSummaryService().build(
        user_id="user-1",
        session_id="session-1",
        previous=None,
        new_messages=MESSAGES,
    )

    assert summary.version == 1
    assert summary.covers_until_message_id == 2
    assert summary.structured_data["pending_tasks"] == ["完成风险测评"]


def test_summary_service_rejects_invalid_llm_output(monkeypatch) -> None:
    monkeypatch.setattr(
        session_summary_service,
        "invoke_llm",
        lambda messages, run_name, prompt_version="v1": SimpleNamespace(content="not-json"),
    )

    with pytest.raises(ValueError, match="summary output"):
        session_summary_service.SessionSummaryService().build(
            user_id="user-1",
            session_id="session-1",
            previous=None,
            new_messages=MESSAGES,
        )


def test_summary_service_rejects_hallucinated_pii(monkeypatch) -> None:
    monkeypatch.setattr(
        session_summary_service,
        "invoke_llm",
        lambda messages, run_name, prompt_version="v1": SimpleNamespace(content='''{
            "summary_text":"用户手机号为13800138000",
            "goals":[],"entities":{},"actions":[],"results":[],"missing_info":[],"pending_tasks":[]
        }'''),
    )

    with pytest.raises(ValueError, match="sensitive content"):
        session_summary_service.SessionSummaryService().build(
            user_id="user-1",
            session_id="session-1",
            previous=None,
            new_messages=MESSAGES,
        )
