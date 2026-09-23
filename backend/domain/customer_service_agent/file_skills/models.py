"""文件 Skill 的元数据、冻结索引和完整加载结果。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SkillMetadata(BaseModel):
    """来自 `SKILL.md` frontmatter 的严格授权元数据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")
    description: str = Field(min_length=20, max_length=2000)
    version: str = Field(pattern=r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
    domain: Literal["telecom", "retail", "shared"]
    capabilities: tuple[str, ...] = Field(min_length=1)
    allowed_agent_types: tuple[str, ...] = Field(min_length=1)
    allowed_tools: tuple[str, ...] = ()
    effect: Literal["read", "write"]


class SkillIndexEntry(BaseModel):
    """启动扫描得到的轻量条目；哈希用于检测冻结后的文件变化。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    metadata: SkillMetadata
    path: Path
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class LoadedSkill(BaseModel):
    """通过权限与完整性校验后可交给领域 Agent 使用的 Skill。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    metadata: SkillMetadata
    body: str
    references: dict[str, str]
    path: Path
    content_hash: str

    def identity(self) -> dict[str, str]:
        """返回写入治理动作和审计轨迹的稳定 Skill 身份。"""

        return {
            "skill_name": self.metadata.name,
            "skill_version": self.metadata.version,
            "skill_content_hash": self.content_hash,
        }
