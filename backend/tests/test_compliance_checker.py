from __future__ import annotations

from domain.customer_service_agent.workflow.nodes import compliance_checker_node as compliance
from domain.customer_service_agent.workflow.nodes.compliance_checker_node import rule_check


def test_rule_check_flags_forbidden_terms():
    result = rule_check("该产品保证收益，零风险。")
    assert result.passed is False
    assert result.risk_level == "high"
    assert result.violations


def test_checker_uses_raw_query_when_sub_results_are_empty(monkeypatch):
    checked: dict[str, str] = {}

    def fake_full_check(content: str) -> compliance.ComplianceResult:
        checked["content"] = content
        return compliance.ComplianceResult(True, "low")

    monkeypatch.setattr(compliance, "full_check", fake_full_check)

    output = compliance.compliance_checker_node({
        "raw_query": "查询订单状态",
        "sub_results": {},
        "draft_source": "llm",
    })

    assert checked["content"] == "查询订单状态"
    assert output["compliance_passed"] is True
    assert output["current_agent"] == "compliance_checker"


def test_checker_preserves_business_pii_in_sub_results_when_blocked(monkeypatch):
    monkeypatch.setattr(
        compliance,
        "full_check",
        lambda _content: compliance.ComplianceResult(False, "high", violations=["blocked"]),
    )

    output = compliance.compliance_checker_node({
        "raw_query": "fallback",
        "sub_results": {
            "answer": "手机号 13812345000",
            "metadata": {"phone": "13812345000"},
        },
        "draft_source": "llm",
    })

    assert "13812345000" in output["sub_results"]["answer"]
    assert output["sub_results"]["metadata"] == {"phone": "13812345000"}
    assert output["compliance_result"]["violations"] == ["blocked"]


def test_full_check_merges_rule_and_llm_results(monkeypatch):
    monkeypatch.setattr(
        compliance,
        "llm_check",
        lambda _content: compliance.ComplianceResult(
            False,
            "medium",
            violations=["需要人工复核"],
            suggestions=["改写回复"],
            sanitized_content="sanitized-by-llm",
        ),
    )

    result = compliance.full_check("普通客服回复")

    assert result.passed is False
    assert result.risk_level == "medium"
    assert result.violations == ["需要人工复核"]
    assert result.suggestions == ["改写回复"]
    assert result.sanitized_content == "普通客服回复"


def test_llm_check_blocks_invalid_json(monkeypatch):
    class Response:
        content = "不是 JSON"

    monkeypatch.setattr(compliance, "invoke_llm", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(compliance, "record_json_parse", lambda *_args, **_kwargs: None)

    result = compliance.llm_check("手机号 13812345000")

    assert result.passed is False
    assert result.risk_level == "high"
    assert result.violations == ["合规审查结果格式无效，已转人工复核"]
    assert "13812345000" in result.sanitized_content


def test_rule_check_allows_full_phone_and_address_output():
    content = "当前默认地址：江苏省南京市栖霞区文艺路9号，手机号：18060815554"

    result = rule_check(content)

    assert result.passed is True
    assert result.sanitized_content == content
