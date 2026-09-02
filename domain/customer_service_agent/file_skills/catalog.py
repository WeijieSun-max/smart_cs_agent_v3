"""扫描、冻结并按需加载版本化文件 Skill。"""

from __future__ import annotations

from pathlib import Path

from domain.customer_service_agent.tools.mcp_server import MCPToolServer

from .integrity import discover_skill_sources, read_skill_source, validate_skill_lock
from .models import LoadedSkill, SkillIndexEntry, SkillMetadata


class FileSkillCatalog:
    """以 ``**/SKILL.md`` 为唯一业务来源的冻结目录。

    启动阶段解析 frontmatter，并校验由说明正文和显式引用共同组成的 bundle
    哈希。正文和引用资料不保留在目录中，而是在选中 Skill 后重新读取。冻结后
    再次校验 bundle 哈希，可防止请求执行被替换的说明或引用文件。
    """

    def __init__(self, root: Path, tool_server: MCPToolServer):
        self.root = root.resolve()
        self._tool_server = tool_server
        self._entries: dict[tuple[str, str], SkillIndexEntry] = {}
        self._active_by_name: dict[str, SkillIndexEntry] = {}
        self._frozen = False

    def scan_and_freeze(self) -> None:
        """扫描全部 Skill、验证唯一性和工具授权，并选出每个名称的最高版本。"""

        if self._frozen:
            raise RuntimeError("skill catalog is already frozen")
        if not self.root.is_dir():
            raise ValueError(f"skill root does not exist: {self.root}")
        sources = discover_skill_sources(self.root)
        validate_skill_lock(self.root, sources)
        for source in sources:
            metadata = source.metadata
            key = (metadata.name, metadata.version)
            if key in self._entries:
                raise ValueError(f"duplicate skill name/version: {metadata.name}@{metadata.version}")
            self._validate_tool_contract(metadata)
            entry = SkillIndexEntry(
                metadata=metadata,
                path=source.path,
                content_hash=source.content_hash,
            )
            self._entries[key] = entry
            current = self._active_by_name.get(metadata.name)
            if current is None or _semver(metadata.version) > _semver(current.metadata.version):
                self._active_by_name[metadata.name] = entry
        self._frozen = True

    def summaries(self) -> list[dict[str, object]]:
        """返回供 Supervisor 选择的轻量索引，不提前暴露正文和引用。"""

        self._require_frozen()
        return [
            {
                "name": entry.metadata.name,
                "version": entry.metadata.version,
                "description": entry.metadata.description,
                "domain": entry.metadata.domain,
                "capabilities": list(entry.metadata.capabilities),
                "effect": entry.metadata.effect,
                "content_hash": entry.content_hash,
            }
            for entry in sorted(self._active_by_name.values(), key=lambda item: item.metadata.name)
        ]

    def select(self, *, capability: str, agent_type: str) -> SkillIndexEntry | None:
        """按能力和 Agent 类型选择唯一活跃 Skill；歧义时拒绝猜测。"""

        self._require_frozen()
        matches = [
            entry for entry in self._active_by_name.values()
            if capability in entry.metadata.capabilities
            and agent_type in entry.metadata.allowed_agent_types
        ]
        if len(matches) > 1:
            raise RuntimeError(f"ambiguous skill capability: {capability}")
        return matches[0] if matches else None

    def load(self, name: str, version: str | None = None, *, agent_type: str) -> LoadedSkill:
        """校验调用方、文件哈希和引用路径后渐进加载 Skill 内容。"""

        self._require_frozen()
        entry = self._active_by_name.get(name) if version is None else self._entries.get((name, version))
        if entry is None:
            raise KeyError(name)
        if agent_type not in entry.metadata.allowed_agent_types:
            raise PermissionError(f"agent type cannot load skill: {agent_type}")
        source = read_skill_source(entry.path)
        if source.content_hash != entry.content_hash:
            raise RuntimeError("skill changed after catalog freeze")
        return LoadedSkill(
            metadata=entry.metadata,
            body=source.body,
            references=source.references,
            path=entry.path,
            content_hash=entry.content_hash,
        )

    def _validate_tool_contract(self, metadata: SkillMetadata) -> None:
        """保证 Skill 只能声明已注册且效果、Agent 归属兼容的工具。"""

        for tool_name in metadata.allowed_tools:
            tool = self._tool_server.get_tool(tool_name)
            if tool is None:
                raise ValueError(f"skill references unknown tool: {tool_name}")
            if metadata.effect == "read" and tool.effect != "read":
                raise ValueError(f"read skill cannot allow write tool: {tool_name}")
            if metadata.allowed_agent_types and tool.allowed_agent_types:
                if not set(metadata.allowed_agent_types).intersection(tool.allowed_agent_types):
                    raise ValueError(f"skill/tool agent contract mismatch: {tool_name}")

    def _require_frozen(self) -> None:
        """阻止在启动校验完成前读取不稳定目录。"""

        if not self._frozen:
            raise RuntimeError("skill catalog is not frozen")


def _semver(value: str) -> tuple[int, int, int]:
    """把已由 Pydantic 校验的三段式语义版本转换为可比较元组。"""

    return tuple(int(part) for part in value.split("."))  # type: ignore[return-value]


_catalog: FileSkillCatalog | None = None


def initialize_catalog(root: Path, tool_server: MCPToolServer) -> FileSkillCatalog:
    """构建、冻结并安装进程级 Skill 目录。"""

    global _catalog
    catalog = FileSkillCatalog(root, tool_server)
    catalog.scan_and_freeze()
    _catalog = catalog
    return catalog


def get_catalog() -> FileSkillCatalog:
    """返回已冻结目录；基础设施尚未初始化时立即失败。"""

    if _catalog is None:
        raise RuntimeError("file skill catalog is not initialized")
    return _catalog
