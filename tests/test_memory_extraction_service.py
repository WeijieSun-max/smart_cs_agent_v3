from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from domain.customer_service_agent.memory.models import MemoryType
from domain.customer_service_agent.service import memory_extraction_service


NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)


def test_extraction_service_returns_only_policy_accepted_candidates(monkeypatch) -> None:
    monkeypatch.setattr(
        memory_extraction_service,
        "invoke_llm",
        lambda messages, run_name, prompt_version="v1": SimpleNamespace(content='''{
          "memories":[
            {"memory_type":"preference","memory_key":"preference.risk","content":"用户明确偏好稳健型产品","confidence":0.9,"structured_data":{}},
            {"memory_type":"fact","memory_key":"fact.phone","content":"用户手机号为13800138000","confidence":0.99,"structured_data":{}}
          ]
        }'''),
    )

    candidates = memory_extraction_service.MemoryExtractionService.default().extract(
        summary_text="用户咨询理财产品",
        messages=[],
        now=NOW,
    )

    assert len(candidates) == 1
    assert candidates[0].memory_type is MemoryType.PREFERENCE


def test_extraction_service_rejects_invalid_json(monkeypatch) -> None:
    monkeypatch.setattr(
        memory_extraction_service,
        "invoke_llm",
        lambda messages, run_name, prompt_version="v1": SimpleNamespace(content="{}"),
    )

    with pytest.raises(ValueError, match="memory extraction output"):
        memory_extraction_service.MemoryExtractionService.default().extract(
            summary_text="summary",
            messages=[],
            now=NOW,
        )

