import asyncio
from dataclasses import FrozenInstanceError

import pytest

from domain.customer_service_agent.tools.mcp_server import MCPToolServer
from domain.customer_service_agent.tools.tool_registry import get_mcp_server
from pkg.exceptions.exception import ToolValidationError


def test_production_tool_discovery_contains_only_telecom_and_retail_tools() -> None:
    names = {tool["name"] for tool in get_mcp_server().list_tools()}

    assert {
        "telecom_get_current_plan",
        "telecom_change_plan",
        "retail_get_order",
        "retail_cancel_order",
        "retail_create_address",
    } <= names
    assert not names.intersection({"order_query", "knowledge_search", "ticket_create", "ticket_query", "risk_check", "user_profile"})


def test_legacy_tool_call_is_not_found() -> None:
    result = asyncio.run(get_mcp_server().call_tool("ticket_create", {}))

    assert result.success is False
    assert result.error_code == "tool.not_found"


def test_every_production_write_tool_is_governed() -> None:
    writes = [tool for tool in get_mcp_server().list_tools() if tool["effect"] == "write"]

    assert writes
    assert all(tool["supportsIdempotency"] for tool in writes)
    assert all(tool["confirmationPolicy"] == "always" for tool in writes)
    assert all(tool["parallelSafe"] is False for tool in writes)


def test_tool_discovery_exposes_governed_effect_metadata() -> None:
    server = get_mcp_server()
    by_name = {tool["name"]: tool for tool in server.list_tools()}

    assert by_name["telecom_get_current_plan"]["effect"] == "read"
    assert by_name["telecom_get_current_plan"]["parallelSafe"] is True
    assert by_name["retail_cancel_order"]["effect"] == "write"
    assert by_name["retail_cancel_order"]["supportsIdempotency"] is True
    assert by_name["retail_cancel_order"]["confirmationPolicy"] == "always"


def test_business_write_cannot_bypass_governed_action_gateway() -> None:
    result = asyncio.run(
        get_mcp_server().call_tool(
            "retail_cancel_order",
            {"order_id": "order-1", "expected_version": 1, "reason": "changed_mind"},
            trusted_context={"user_id": "user-1"},
        )
    )

    assert result.success is False
    assert result.error_code == "tool.validation"


def test_validate_arguments_exposes_business_schema_without_execution() -> None:
    server = get_mcp_server()
    arguments = {"order_id": "order-1", "expected_version": 1, "reason": "changed_mind"}

    server.validate_arguments("retail_cancel_order", arguments)
    with pytest.raises(ToolValidationError):
        server.validate_arguments("retail_cancel_order", {**arguments, "user_id": "forged"})


def test_register_requires_valid_effect_and_idempotency_metadata() -> None:
    server = MCPToolServer()

    async def handler(_trusted_context=None):
        return "ok"

    with pytest.raises(TypeError):
        server.register("missing", "missing metadata", {"type": "object"})(handler)
    with pytest.raises(TypeError):
        server.register(
            "missing_idempotency",
            "missing idempotency metadata",
            {"type": "object"},
            effect="read",
        )(handler)
    with pytest.raises(ValueError, match="effect"):
        server.register(
            "invalid",
            "invalid effect",
            {"type": "object"},
            effect="external",
            supports_idempotency=False,
        )(handler)


def test_register_rejects_duplicate_name_without_changing_original_definition() -> None:
    server = MCPToolServer()

    @server.register(
        "stable_tool",
        "original definition",
        {"type": "object", "properties": {}},
        effect="read",
        supports_idempotency=False,
    )
    async def original(_trusted_context=None):
        return "original"

    with pytest.raises(ValueError, match="duplicate"):
        @server.register(
            "stable_tool",
            "replacement definition",
            {"type": "object", "properties": {"value": {"type": "string"}}},
            effect="write",
            supports_idempotency=True,
        )
        async def replacement(value: str = "", _trusted_context=None):
            return value

    definition = server.get_tool("stable_tool")
    assert definition is not None
    assert definition.description == "original definition"
    assert definition.effect == "read"
    assert definition.supports_idempotency is False
    assert definition.handler is original


def test_tool_definitions_and_schemas_are_detached_snapshots() -> None:
    server = MCPToolServer()
    source_schema = {
        "type": "object",
        "properties": {"count": {"type": "integer", "minimum": 1}},
    }

    @server.register(
        "counter",
        "validated counter",
        source_schema,
        effect="read",
        supports_idempotency=False,
    )
    async def counter(count: int = 1, _trusted_context=None):
        return count

    source_schema["properties"]["count"]["minimum"] = -100
    definition = server.get_tool("counter")
    assert definition is not None
    with pytest.raises(FrozenInstanceError):
        definition.effect = "write"
    definition.input_schema["properties"]["count"]["minimum"] = -100
    listed = server.list_tools()
    listed[0]["inputSchema"]["properties"]["count"]["minimum"] = -100

    fresh_definition = server.get_tool("counter")
    assert fresh_definition is not None
    assert fresh_definition.effect == "read"
    assert fresh_definition.input_schema["properties"]["count"]["minimum"] == 1
    assert server.list_tools()[0]["inputSchema"]["properties"]["count"]["minimum"] == 1
    with pytest.raises(ToolValidationError):
        server.validate_arguments("counter", {"count": 0})


def test_validate_arguments_accepts_business_contact_details() -> None:
    server = MCPToolServer()

    @server.register(
        "echo_text",
        "echo text",
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        effect="read",
        supports_idempotency=False,
    )
    async def echo_text(text: str, _trusted_context=None):
        return text

    server.validate_arguments("echo_text", {"text": "call 13800138000"})


def test_call_tool_always_injects_trusted_context_keyword() -> None:
    server = MCPToolServer()
    received_contexts = []

    @server.register(
        "required_context",
        "handler requires trusted context keyword",
        {"type": "object", "properties": {}},
        effect="read",
        supports_idempotency=False,
    )
    async def required_context(_trusted_context):
        received_contexts.append(_trusted_context)
        return "ok"

    without_context = asyncio.run(server.call_tool("required_context", {}))
    supplied_context = {"operation_key": "skill-action:one"}
    with_context = asyncio.run(
        server.call_tool(
            "required_context",
            {},
            trusted_context=supplied_context,
        )
    )

    assert without_context.success is True
    assert with_context.success is True
    assert received_contexts[0] is None
    assert received_contexts[1] == supplied_context
    assert received_contexts[1] is not supplied_context
