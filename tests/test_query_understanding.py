from domain.customer_service_agent.orchestration import query_understanding
from domain.customer_service_agent.orchestration.models import AgentAssignment, SupervisorDecision
from domain.customer_service_agent.workflow.entity.chat_state import create_chat_state


def test_understanding_is_projection_of_llm_supervisor(monkeypatch) -> None:
    async def decide(*_args, **_kwargs):
        return SupervisorDecision(
            action="dispatch",
            standalone_query="申请退回2026-08-24购买的第二部手机",
            assignments=(AgentAssignment(
                task_id="T1",
                agent="retail_agent",
                objective="申请退货",
                capability="request_return",
                arguments={
                    "product_query": "手机",
                    "ordinal": 2,
                    "start_date": "2026-08-24",
                    "end_date": "2026-08-25",
                    "quantity": 1,
                },
            ),),
            confidence=0.96,
        )

    monkeypatch.setattr(query_understanding, "decide_next_step", decide)
    state = create_chat_state("user-1", "session-1", "把昨天买的第二个手机退掉")
    state["current_time"] = "2026-08-25"

    result = query_understanding.understand_query(state)

    assert result.source == "llm"
    assert result.capabilities == ("request_return",)
    assert result.entities["ordinal"] == 2
    assert result.temporal_range == {"start": "2026-08-24", "end": "2026-08-25"}


def test_finish_decision_projects_to_fallback(monkeypatch) -> None:
    async def decide(*_args, **_kwargs):
        return SupervisorDecision(
            action="finish",
            standalone_query="你好",
            response="你好，请问需要什么帮助？",
            confidence=0.99,
        )

    monkeypatch.setattr(query_understanding, "decide_next_step", decide)

    result = query_understanding.understand_query(
        create_chat_state("user-1", "session-1", "你好")
    )

    assert result.source == "llm"
    assert result.domains == ("fallback",)
    assert result.capabilities == ("fallback",)
