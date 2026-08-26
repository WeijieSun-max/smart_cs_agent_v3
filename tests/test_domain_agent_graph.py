import asyncio

from domain.customer_service_agent.orchestration.models import AgentResult, QueryUnderstandingResult, RouteDecision, TaskPlan, TaskSpec
from domain.customer_service_agent.workflow.entity.chat_state import create_chat_state
from domain.customer_service_agent.workflow.nodes import supervisor_graph_nodes


class ConcurrencyTracker:
    def __init__(self) -> None:
        self.started = 0
        self.inflight = 0
        self.max_inflight = 0
        self.ready = asyncio.Event()


class FakeDomainGraph:
    def __init__(self, domain: str, tracker: ConcurrencyTracker) -> None:
        self.domain = domain
        self.tracker = tracker
        self.received = None

    async def ainvoke(self, state):
        self.received = state
        self.tracker.started += 1
        self.tracker.inflight += 1
        self.tracker.max_inflight = max(self.tracker.max_inflight, self.tracker.inflight)
        if self.tracker.started == 2:
            self.tracker.ready.set()
        await asyncio.wait_for(self.tracker.ready.wait(), timeout=1)
        await asyncio.sleep(0)
        self.tracker.inflight -= 1
        task = state["tasks"][0]
        return {"results": [AgentResult(
            task_id=task["task_id"],
            status="succeeded",
            user_fragment=f"{self.domain}-result",
        ).model_dump(mode="json")]}


def _cross_domain_state():
    state = create_chat_state("user-1", "session-1", "查询套餐和订单")
    understanding = QueryUnderstandingResult(
        standalone_query="查询当前套餐和订单 order_id:order-1",
        domains=("telecom", "retail"),
        capabilities=("current_plan", "order_query"),
        entities={"order_id": "order-1"},
        confidence=0.98,
        source="llm",
    )
    route = RouteDecision(
        domains=("telecom", "retail"),
        capabilities=("current_plan", "order_query"),
        confidence=0.98,
        composite=True,
    )
    plan = TaskPlan(tasks=(
        TaskSpec(task_id="T1", domain="telecom", capability="current_plan"),
        TaskSpec(task_id="T2", domain="retail", capability="order_query", arguments={"order_id": "order-1"}),
    ))
    state.update(
        query_understanding=understanding.model_dump(mode="json"),
        route_decision=route.model_dump(mode="json"),
        task_plan=plan.model_dump(mode="json"),
    )
    return state


def test_domain_dispatch_runs_telecom_and_retail_subgraphs_concurrently(monkeypatch) -> None:
    tracker = ConcurrencyTracker()
    telecom = FakeDomainGraph("telecom", tracker)
    retail = FakeDomainGraph("retail", tracker)
    monkeypatch.setattr(supervisor_graph_nodes, "telecom_agent_graph", telecom)
    monkeypatch.setattr(supervisor_graph_nodes, "retail_agent_graph", retail)

    result = asyncio.run(supervisor_graph_nodes.domain_dispatch_node(_cross_domain_state()))

    assert tracker.max_inflight == 2
    assert list(result["domain_agent_results"]) == ["telecom", "retail"]
    assert [task["domain"] for task in telecom.received["tasks"]] == ["telecom"]
    assert [task["domain"] for task in retail.received["tasks"]] == ["retail"]


def test_result_aggregator_preserves_task_plan_order() -> None:
    state = _cross_domain_state()
    state["domain_agent_results"] = {
        "retail": [AgentResult(task_id="T2", status="succeeded", user_fragment="retail-result").model_dump(mode="json")],
        "telecom": [AgentResult(task_id="T1", status="succeeded", user_fragment="telecom-result").model_dump(mode="json")],
    }

    result = supervisor_graph_nodes.result_aggregator_node(state)

    assert result["intent"] == "composite"
    assert result["sub_results"]["supervisor"] == "telecom-result\n\nretail-result"
    assert list(result["task_results"]) == ["T1", "T2"]


def test_pending_action_route_bypasses_understanding() -> None:
    state = create_chat_state("user-1", "session-1", "确认")
    state["pending_action_handled"] = True

    assert supervisor_graph_nodes.pending_action_route(state) == "handled"
