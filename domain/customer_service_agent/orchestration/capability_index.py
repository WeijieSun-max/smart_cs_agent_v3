"""从工具注册表派生 Supervisor 和领域 Agent 共用的能力索引。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from domain.customer_service_agent.tools.tool_registry import get_mcp_server

# 能力词表的唯一事实来源：工具注册表（business_tools.py 的 @server.register 元数据）。
# Supervisor 与领域 Agent 只从这里派生或校验能力，不维护关键词词表。

# 非工具能力：不通过注册表工具执行，而是 RAG 或合成能力。
NON_TOOL_CAPABILITIES = frozenset({"telecom_troubleshooting", "retail_policy", "fallback"})

# 内部能力：工具注册表声明、但不作为用户可路由意图暴露给理解层的词（由执行层内部使用）。
INTERNAL_CAPABILITIES = frozenset({"plan_catalog", "order_resolution", "payment_query"})


@dataclass(frozen=True)
class ToolFacet:
    """路由所需的最小工具元数据，避免编排层依赖完整处理器定义。"""

    name: str
    domain: str
    effect: str
    capabilities: tuple[str, ...]
    allowed_agent_types: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityIndex:
    """按读写效果和 capability 聚合后的不可变工具视图。"""

    tools: tuple[ToolFacet, ...]
    capabilities: frozenset[str]  # 注册表全部能力（不含 fallback）
    write_capabilities: frozenset[str]
    read_capabilities: frozenset[str]
    read_tools_by_capability: dict[str, tuple[str, ...]]
    write_tools_by_capability: dict[str, tuple[str, ...]]

    def allowed_read_tools(self, domain: str, agent_type: str) -> frozenset[str]:
        """领域 Agent 只读工具白名单，排除仅服务于写操作报价/预检的工具。"""
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
    """读取已注册工具并一次性构建能力、读工具和写工具反向索引。"""

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
    """惰性返回进程级索引；工具注册须在首次调用前完成。"""

    global _index
    if _index is None:
        _index = build_capability_index()
    return _index


def validate_capability_contracts() -> list[str]:
    """返回工具效果、领域、能力和 Agent 归属中的全部契约问题。"""

    index = get_capability_index()
    problems: list[str] = []

    for tool in index.tools:
        if tool.effect not in {"read", "write"}:
            problems.append(f"tool has invalid effect: {tool.name}={tool.effect}")
        if tool.domain not in {"telecom", "retail", "shared"}:
            problems.append(f"tool has invalid domain: {tool.name}={tool.domain}")
        if not tool.capabilities:
            problems.append(f"tool has no capabilities: {tool.name}")
        expected_agent = f"{tool.domain}_agent"
        if tool.domain in {"telecom", "retail"} and expected_agent not in tool.allowed_agent_types:
            problems.append(f"tool is not assigned to {expected_agent}: {tool.name}")

    return problems
