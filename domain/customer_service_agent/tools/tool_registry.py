"""创建唯一工具服务器，并通过导入处理器完成启动期注册。"""

from __future__ import annotations

from domain.customer_service_agent.tools.mcp_server import MCPToolServer


server = MCPToolServer()


def get_mcp_server() -> MCPToolServer:
    """返回进程级工具注册表。"""

    return server


from . import business_tools as _business_tools  # noqa: E402,F401
