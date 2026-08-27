from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from domain.customer_service_agent.tools.tool_registry import get_mcp_server

# 能力词表的唯一事实来源：工具注册表（business_tools.py 的 @server.register 元数据）。
# Supervisor、领域 Agent 与兼容执行器只从这里派生或校验能力，不维护关键词词表。

# 非工具能力：不通过注册表工具执行，而是 RAG 或合成能力。
NON_TOOL_CAPABILITIES = frozenset({"telecom_troubleshooting", "retail_policy", "fallback"})

# 内部能力：工具注册表声明、但不作为用户可路由意图暴露给理解层的词（由执行层内部使用）。
INTERNAL_CAPABILITIES = frozenset({"plan_catalog", "order_resolution", "payment_query"})


@dataclass(frozen=True)
class ToolFacet:
    name: str
    domain: str
    effect: str
    capabilities: tuple[str, ...]
    allowed_agent_types: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityIndex:
    tools: tuple[ToolFacet, ...]
    capabilities: frozenset[str]  # 注册表全部能力（不含 fallback）
    write_capabilities: frozenset[str]
    read_capabilities: frozenset[str]
    read_tools_by_capability: dict[str, tuple[str, ...]]
    write_tools_by_capability: dict[str, tuple[str, ...]]

    def allowed_read_tools(self, domain: str, agent_type: str) -> frozenset[str]:
        """ReAct 只读工具白名单：域内 read 工具，且能力不是纯写意图（排除报价/预检工具）。"""
        write = self.write_capabilities
        result: set[str] = set()
        for tool in self.tools:
            if tool.effect != "read" or tool.domain != domain:
                continue
            if tool.allowed_agent_types and agent_type not in tool.allowed_agent_types:
                continue
            # 排除“纯写意图的只读报价/预检”工具（如 telecom_quote_plan_change，capability 是写能力）。
            if tool.capabilities and set(tool.capabilities) <= write:
                continue
            result.add(tool.name)
        return frozenset(result)

    def routable_capabilities(self) -> frozenset[str]:
        """可作为用户意图暴露的能力 = 注册表能力 - 内部能力 + 非工具能力。"""
        return (self.capabilities - INTERNAL_CAPABILITIES) | NON_TOOL_CAPABILITIES


def build_capability_index(server: Any | None = None) -> CapabilityIndex:
    server = server or get_mcp_server()
    tools: list[ToolFacet] = []
    capabilities: set[str] = set()
    write_capabilities: set[str] = set()
    read_capabilities: set[str] = set()
    read_tools_by_capability: dict[str, list[str]] = {}
    write_tools_by_capability: dict[str, list[str]] = {}

    for item in server.list_tools():
        name = str(item["name"])
        domain = str(item.get("domain") or "shared")
        effect = str(item.get("effect") or "read")
        caps = tuple(str(cap) for cap in (item.get("capabilities") or ()))
        agent_types = tuple(str(agent) for agent in (item.get("allowedAgentTypes") or ()))
        tools.append(ToolFacet(
            name=name,
            domain=domain,
            effect=effect,
            capabilities=caps,
            allowed_agent_types=agent_types,
        ))
        capabilities.update(caps)
        if effect == "write":
            write_capabilities.update(caps)
            for cap in caps:
                write_tools_by_capability.setdefault(cap, []).append(name)
        elif effect == "read":
            read_capabilities.update(caps)
            for cap in caps:
                read_tools_by_capability.setdefault(cap, []).append(name)

    return CapabilityIndex(
        tools=tuple(tools),
        capabilities=frozenset(capabilities),
        write_capabilities=frozenset(write_capabilities),
        read_capabilities=frozenset(read_capabilities),
        read_tools_by_capability={cap: tuple(names) for cap, names in read_tools_by_capability.items()},
        write_tools_by_capability={cap: tuple(names) for cap, names in write_tools_by_capability.items()},
    )


_index: CapabilityIndex | None = None


def get_capability_index() -> CapabilityIndex:
    global _index
    if _index is None:
        _index = build_capability_index()
    return _index


def reset_capability_index() -> None:
    global _index
    _index = None


def validate_capability_contracts() -> list[str]:
    """Validate tool mappings; supervisor vocabulary is derived dynamically."""
    index = get_capability_index()
    problems: list[str] = []

    from domain.customer_service_agent.orchestration.tool_scheduler import CAPABILITY_TO_TOOL

    # CAPABILITY_TO_TOOL 每个直连映射都必须指向声明了该能力的 read 工具。
    for cap, tool_name in CAPABILITY_TO_TOOL.items():
        read_tools = index.read_tools_by_capability.get(cap, ())
        if tool_name not in read_tools:
            problems.append(f"CAPABILITY_TO_TOOL[{cap}]={tool_name} is not a read tool for that capability")

    # 每个写能力都必须有至少一个 write 工具（保证写意图可执行、可治理）。
    for cap in index.write_capabilities:
        if cap not in index.write_tools_by_capability:
            problems.append(f"write capability has no write tool: {cap}")

    return problems
