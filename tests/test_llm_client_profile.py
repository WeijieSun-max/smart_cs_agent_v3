from types import SimpleNamespace

from domain.shared.llm.llm_service import ModelProfile
from infra.llm import openai_compatible_client


def test_qwen_profile_disables_thinking_for_bounded_structured_outputs(monkeypatch) -> None:
    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(openai_compatible_client, "ChatOpenAI", fake_chat_openai)
    settings = SimpleNamespace(qwen_api_key="test-key")
    profile = ModelProfile(
        name="supervisor",
        model="qwen-test",
        base_url="https://example.invalid/v1",
        max_tokens=1200,
    )

    openai_compatible_client.create_profile_chat_model(settings, profile)

    assert captured["extra_body"] == {"enable_thinking": False}
    assert captured["max_tokens"] == 1200
