import asyncio

from domain.customer_service_agent.orchestration.models import AgentAssignment, AgentResult, SupervisorDecision
from domain.customer_service_agent.workflow.entity.chat_state import create_chat_state
from domain.customer_service_agent.workflow.nodes import supervisor_graph_nodes


class ConcurrencyTracker:
    def __init__(self) -> None:
        self.started = 0
        self.inflight = 0
        self.max_inflight = 0
        self.ready = asyncio.Event()


class FakeDomainGraph:
    def __init__(self, agent: str, tracker: ConcurrencyTracker) -> None:
        self.agent = agent
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
        self.tracker.inflight -= 1
        assignment = state["assignment"]
        return {"result": AgentResult(
            task_id=assignment["task_id"],
            agent=assignment["agent"],
            status="succeeded",
            user_fragment=f"{self.agent}-result",
        ).model_dump(mode="json")}


class SequentialDomainGraph:
    def __init__(self, result: AgentResult, calls: list[str]) -> None:
        self.result = result
        self.calls = calls
        self.received = None

    async def ainvoke(self, state):
        self.received = state
        self.calls.append(self.result.task_id)
        return {"result": self.result.model_dump(mode="json")}


def _cross_domain_state():
    state = create_chat_state("user-1", "session-1", "查询套餐和订单")
    assignments = (
        AgentAssignment(
            task_id="T1",
            agent="telecom_agent",
            objective="查询当前套餐",
            capability="current_plan",
        ),
        AgentAssignment(
            task_id="T2",
            agent="retail_agent",
            objective="查询订单",
            capability="order_query",
            arguments={"order_id": "order-1"},
        ),
    )
    state["agent_assignments"] = [item.model_dump(mode="json") for item in assignments]
    return state


def _read_write_state():
    state = create_chat_state("user-1", "session-rw", "查询订单并变更套餐")
    assignments = (
        AgentAssignment(
            task_id="T1",
            agent="telecom_agent",
            objective="生成套餐变更待确认提案",
            capability="plan_change",
            arguments={"line_id": "L1", "plan_id": "P2"},
        ),
        AgentAssignment(
            task_id="T2",
            agent="retail_agent",
            objective="查询订单",
            capability="order_query",
            arguments={"order_id": "order-1"},
        ),
    )
    state["agent_assignments"] = [item.model_dump(mode="json") for item in assignments]
    return state


def test_domain_dispatch_runs_independent_subgraphs_concurrently(monkeypatch) -> None:
    tracker = ConcurrencyTracker()
    telecom = FakeDomainGraph("telecom", tracker)
    retail = FakeDomainGraph("retail", tracker)
    monkeypatch.setattr(supervisor_graph_nodes, "telecom_agent_graph", telecom)
    monkeypatch.setattr(supervisor_graph_nodes, "retail_agent_graph", retail)

    result = asyncio.run(supervisor_graph_nodes.domain_dispatch_node(_cross_domain_state()))

    assert tracker.max_inflight == 2
    assert list(result["task_results"]) == ["T1", "T2"]
    assert list(result["domain_agent_results"]) == ["telecom_agent", "retail_agent"]
    assert telecom.received["assignment"]["agent"] == "telecom_agent"
    assert retail.received["assignment"]["agent"] == "retail_agent"


def test_domain_dispatch_runs_reads_with_one_write_proposal_concurrently(monkeypatch) -> None:
    tracker = ConcurrencyTracker()
    telecom = FakeDomainGraph("telecom", tracker)
    retail = FakeDomainGraph("retail", tracker)
    monkeypatch.setattr(supervisor_graph_nodes, "telecom_agent_graph", telecom)
    monkeypatch.setattr(supervisor_graph_nodes, "retail_agent_graph", retail)

    result = asyncio.run(supervisor_graph_nodes.domain_dispatch_node(_read_write_state()))

    assert tracker.max_inflight == 2
    assert list(result["task_results"]) == ["T1", "T2"]


def test_domain_dispatch_passes_declared_cross_agent_dependency_results(monkeypatch) -> None:
    calls = []
    knowledge_result = AgentResult(
        task_id="K1",
        agent="knowledge_agent",
        status="succeeded",
        facts={"rag": {"grounded": True}},
        user_fragment="退货政策要求订单已送达。[1]",
    )
    retail_result = AgentResult(
        task_id="R1",
        agent="retail_agent",
        status="succeeded",
        user_fragment="订单符合退货状态要求。",
    )
    knowledge = SequentialDomainGraph(knowledge_result, calls)
    retail = SequentialDomainGraph(retail_result, calls)
    monkeypatch.setattr(supervisor_graph_nodes, "knowledge_agent_graph", knowledge)
    monkeypatch.setattr(supervisor_graph_nodes, "retail_agent_graph", retail)

    state = create_chat_state("user-1", "session-dependent", "判断订单是否符合退货政策")
    state["agent_assignments"] = [
        AgentAssignment(
            task_id="K1",
            agent="knowledge_agent",
            objective="检索退货政策",
            capability="retail_policy",
        ).model_dump(mode="json"),
        AgentAssignment(
            task_id="R1",
            agent="retail_agent",
            objective="结合政策检查订单状态",
            capability="order_query",
            dependencies=("K1",),
            arguments={"order_id": "order-1"},
        ).model_dump(mode="json"),
    ]

    result = asyncio.run(supervisor_graph_nodes.domain_dispatch_node(state))

    assert calls == ["K1", "R1"]
    assert list(result["task_results"]) == ["K1", "R1"]
    assert knowledge.received["dependency_results"] == {}
    assert retail.received["dependency_results"] == {
        "K1": knowledge_result.model_dump(mode="json")
    }


def test_supervisor_route_uses_typed_llm_action() -> None:
    state = create_chat_state("user-1", "session-1", "你好")
    state["supervisor_decision"] = SupervisorDecision(
        action="finish",
        standalone_query="你好",
        response="你好",
    ).model_dump(mode="json")

    assert supervisor_graph_nodes.supervisor_route(state) == "respond"
