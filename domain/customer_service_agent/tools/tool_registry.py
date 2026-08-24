from __future__ import annotations

from domain.customer_service_agent.tools.mcp_server import MCPToolServer


server = MCPToolServer()


def get_mcp_server() -> MCPToolServer:
    return server


from . import business_tools as _business_tools  # noqa: E402,F401
