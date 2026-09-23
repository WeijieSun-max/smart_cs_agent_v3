from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from domain.shared.llm import llm_service
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


def test_invoke_llm_binds_native_tools_before_invocation(monkeypatch) -> None:
    captured = {}

    class BoundRunnable:
        def invoke(self, messages, config):
            captured["messages"] = messages
            captured["config"] = config
            return "native-result"

    class Client:
        def bind_tools(self, tools):
            captured["tools"] = tools
            return BoundRunnable()

    monkeypatch.setattr(llm_service, "_acquire_slot", lambda: None)
    monkeypatch.setattr(llm_service, "get_llm_client", lambda **_kwargs: Client())
    native_tools = [{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "lookup data",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    messages = [HumanMessage(content="lookup")]

    result = llm_service.invoke_llm(
        messages,
        run_name="retail.agent",
        prompt_version="native-v1",
        tools=native_tools,
    )

    assert result == "native-result"
    assert captured["tools"] == native_tools
    assert captured["tools"] is not native_tools
    assert captured["messages"] is messages
    assert captured["config"]["metadata"]["prompt_version"] == "native-v1"
