"""对响应草稿执行确定性规则和可选 LLM 双层合规审查。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, StrictBool, ValidationError

from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.llm.llm_service import invoke_llm
from pkg.llm import parse_json_object
from pkg.telemetry import record_json_parse

FORBIDDEN_TERMS = ["保证收益", "稳赚不赔", "零风险", "保本保息", "最高收益", "承诺回报", "内部消息", "内幕"]

COMPLIANCE_SYSTEM_PROMPT = """你是一个金融/电商客服合规审查Agent。
请检查回复是否包含违规金融用语、越权承诺、歧视或侮辱性内容。
用户主动提供、或当前用户有权读取的姓名、手机号、邮箱和收货地址属于正常业务数据，允许完整输出，不应判为违规。
只返回JSON：{"passed": true, "risk_level": "low|medium|high|critical", "violations": [], "suggestions": []}
"""


@dataclass
class ComplianceResult:
    """合并规则和模型审查后的内部合规结论。"""

    passed: bool
    risk_level: str
    violations: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    sanitized_content: str = ""


class ComplianceDecision(BaseModel):
    """合规模型必须满足的严格结构化输出。"""

    passed: StrictBool
    risk_level: Literal["low", "medium", "high", "critical"]
    violations: list[str] = Field(default_factory=list, max_length=100)
    suggestions: list[str] = Field(default_factory=list, max_length=100)


def compliance_checker_node(state: ChatState) -> dict:
    """按草稿来源选择审查强度，并写回合规状态。

    确定性模板只运行不可绕过规则；任何含 LLM 内容的草稿运行规则与独立模型
    双检。解析失败按高风险阻断，不能因审查器故障而默认放行。
    """

    content = _content_from_state(state)
    # 风险分级：确定性模板草稿（工具事实拼装）只走规则层（违禁词/PII 检测），
    # 跳过 LLM 审查以省一次调用；LLM 生成的草稿（RAG/ReAct/合成）仍需规则 + LLM 双重审查。
    if state.get("draft_source", "deterministic") == "llm":
        result = full_check(content)
    else:
        result = rule_check(content)
    return _compliance_state_update(state, result)


def full_check(content: str) -> ComplianceResult:
    """先执行规则否决，高风险时短路，否则合并 LLM 结论。"""

    rule_result = rule_check(content)
    if not rule_result.passed and rule_result.risk_level in {"high", "critical"}:
        return rule_result
    return _merge_compliance_results(rule_result, llm_check(content))


def rule_check(content: str) -> ComplianceResult:
    """扫描确定性禁止用语并生成高风险违规列表。"""

    violations = []
    for term in FORBIDDEN_TERMS:
        if term in content:
            violations.append(f"包含违规金融用语: {term}")
    if not violations:
        return ComplianceResult(True, "low", sanitized_content=content)
    return ComplianceResult(False, "high", violations=violations, sanitized_content=content)


def llm_check(content: str) -> ComplianceResult:
    """请求独立合规模型审查；非法或缺失 JSON 一律阻断。"""

    response = invoke_llm([
        SystemMessage(content=COMPLIANCE_SYSTEM_PROMPT),
        HumanMessage(content=f"请审查以下客服回复内容：\n\n{content}"),
    ], run_name="compliance.review")
    parsed = parse_json_object(str(response.content))
    if parsed is None:
        record_json_parse("compliance.review", False)
        return _blocked_parse_failure(content)
    try:
        decision = ComplianceDecision.model_validate(parsed)
    except ValidationError:
        record_json_parse("compliance.review", False)
        return _blocked_parse_failure(content)
    record_json_parse("compliance.review", True)
    return ComplianceResult(
        passed=decision.passed,
        risk_level=decision.risk_level,
        violations=decision.violations,
        suggestions=decision.suggestions,
        sanitized_content=content,
    )


def _content_from_state(state: ChatState) -> str:
    """优先审查完整草稿，兼容旧状态时再拼接子结果。"""

    draft = state.get("draft_response") or ""
    if draft.strip():
        return draft
    content = "\n".join(
        result
        for result in state.get("sub_results", {}).values()
        if isinstance(result, str)
    )
    return content if content.strip() else state["raw_query"]


def _compliance_state_update(state: ChatState, result: ComplianceResult) -> dict:
    """把合规结论转换为主图状态增量。"""

    return {
        "compliance_passed": result.passed,
        "compliance_result": {
            "passed": result.passed,
            "risk_level": result.risk_level,
            "violations": result.violations,
            "suggestions": result.suggestions,
        },
        "sub_results": dict(state.get("sub_results", {})),
        "draft_response": state.get("draft_response", "") if result.passed else result.sanitized_content,
        "current_agent": "compliance_checker",
        "node_logs": [f"合规审查完成：{result.risk_level}"],
    }


def _merge_compliance_results(rule_result: ComplianceResult, llm_result: ComplianceResult) -> ComplianceResult:
    """采用逻辑与合并通过状态，并保留两层中的最高风险。"""

    risk_priority = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    return ComplianceResult(
        passed=rule_result.passed and llm_result.passed,
        risk_level=max(
            rule_result.risk_level,
            llm_result.risk_level,
            key=lambda item: risk_priority.get(item, 0),
        ),
        violations=rule_result.violations + llm_result.violations,
        suggestions=llm_result.suggestions,
        sanitized_content=rule_result.sanitized_content,
    )


def _blocked_parse_failure(content: str) -> ComplianceResult:
    """构造审查输出不可解析时的关闭式高风险结果。"""

    return ComplianceResult(
        False,
        "high",
        violations=["合规审查结果格式无效，已转人工复核"],
        sanitized_content=content,
    )
