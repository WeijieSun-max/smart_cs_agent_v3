"""客服主图的共享状态及单轮初始值构造器。"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState


def merge_node_logs(existing: list[str] | None, new: list[str]) -> list[str]:
    """追加节点日志；节点发出唯一 RESET 标记时清空旧日志。"""
    if new == ["RESET"]:
        return []
    return [*(existing or []), *new]


# 继承 MessagesState，因此 ChatState 自动包含 messages（对话消息历史）字段。
class ChatState(MessagesState):
    """贯穿一次客服轮次的类型化 LangGraph 状态。

    字段分为请求身份、只读会话上下文、Supervisor/Agent 中间结果、治理动作、
    合规草稿和最终响应。节点应返回最小状态增量，避免并发分支覆盖无关字段。
    conversation_context 始终是不可信参考，不能作为授权或当前业务事实。
    """

    user_id: str | None  # 当前用户的唯一标识；未登录或匿名访问时可为空。
    session_id: str  # 会话唯一标识，用于关联同一轮多轮对话。
    current_time: str  # 当前请求的日期上下文，供需要时间信息的节点使用。
    turn_id: str  # 单次请求/对话轮次的唯一标识，粒度比 session_id 更细。
    raw_query: str  # 用户本轮输入的原始问题，未经改写或处理。
    memory_packet: dict[str, Any] | None  # 长期记忆检索结果及其相关元数据。
    conversation_context: dict[str, Any]  # 结构化的 summary、recent_messages 与 memories，仅作不可信参考数据。
    user_message_persisted: bool  # 当前用户消息是否已成功写入持久化存储，避免重复保存。
    normalized_query: str  # 查询改写、指代消解后的独立问题；业务事实仍需由数据库验证。
    domain_agent_results: dict[str, list[dict[str, Any]]]  # Knowledge/Telecom/Retail Agent 的结构化结果。
    intent: str | None  # 识别出的用户意图，用于路由到对应处理流程。
    sub_results: dict[str, Any]  # 子任务或子节点的中间处理结果集合。
    compliance_passed: bool  # 合规校验是否通过；决定是否允许正常输出。
    compliance_result: dict[str, Any]  # 合规校验的详细结果，例如风险类型和拦截原因。
    draft_response: str  # AgentResult 聚合后的完整草稿，必须先合规再发送。
    draft_source: str  # 草稿来源：deterministic（模板/工具事实）或 llm（RAG/ReAct/合成）。用于合规风险分级。
    final_response: str  # 工作流最终生成、准备返回给用户的回复。
    current_agent: str  # 当前正在执行或最近执行的智能体/节点名称。
    node_logs: Annotated[list[str], merge_node_logs]  # 节点执行日志；通过 merge_node_logs 进行累加或重置。
    skill_selection: dict[str, Any] | None  # 技能选择阶段的结果，包括命中的技能及选择依据。
    skill_result: dict[str, Any] | None  # 被调用技能的执行结果。
    request_id: str | None  # 外部请求追踪 ID，用于链路追踪与问题排查。
    identity_source: str  # 用户身份信息的来源，例如请求体、令牌或系统注入。
    auth_strength: str  # 身份认证强度或可信等级，供权限与风险判断使用。
    task_results: dict[str, Any]  # Supervisor 分配的各 Agent 任务结果集合。
    supervisor_round: int  # 当前轮次内 Supervisor 已发起的调度批次数，最多三轮。
    supervisor_decision: dict[str, Any] | None  # Supervisor LLM 本轮结构化决策。
    agent_assignments: list[dict[str, Any]]  # 当前待执行的结构化 Agent 任务。
    agent_assignment_history: list[dict[str, Any]]  # 本轮已调度任务，供复核、评测和审计。
    active_action: dict[str, Any] | None  # 当前会话待确认治理动作的只读摘要。
    supervisor_response: str  # Supervisor 最终回答或澄清问题。
    supervisor_response_source: str  # llm 或 deterministic，供合规层选择审查强度。

# * 之后的参数为仅限关键字参数，调用时必须以参数名传入。
def create_chat_state(
    user_id: str | None,
    session_id: str | None,
    query: str,
    *,
    turn_id: str = "untracked",
    memory_packet: dict[str, Any] | None = None,
    conversation_context: dict[str, Any] | None = None,
    user_message_persisted: bool = False,
) -> ChatState:
    """为新一轮请求创建完整、无共享可变默认值的初始状态。"""

    resolved_session_id = session_id or uuid.uuid4().hex
    return {
        "user_id": user_id,
        "session_id": resolved_session_id,
        "current_time": datetime.now().strftime("%Y-%m-%d"),
        "turn_id": turn_id,
        "raw_query": query,
        "memory_packet": memory_packet,
        "conversation_context": conversation_context or {
            "summary": "",
            "recent_messages": [],
            "memories": [],
        },
        "user_message_persisted": user_message_persisted,
        "messages": [HumanMessage(content=query)],
        "normalized_query": query,
        "domain_agent_results": {},
        "intent": None,
        "sub_results": {},
        "compliance_passed": True,
        "compliance_result": {},
        "draft_response": "",
        "draft_source": "deterministic",
        "final_response": "",
        "current_agent": "",
        "node_logs": [],
        "skill_selection": None,
        "skill_result": None,
        "request_id": None,
        "identity_source": "request_body",
        "auth_strength": "unverified_frontend",
        "task_results": {},
        "supervisor_round": 0,
        "supervisor_decision": None,
        "agent_assignments": [],
        "agent_assignment_history": [],
        "active_action": None,
        "supervisor_response": "",
        "supervisor_response_source": "deterministic",
    }
