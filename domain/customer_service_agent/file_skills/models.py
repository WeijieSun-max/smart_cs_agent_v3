from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SkillMetadata(BaseModel):
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
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    metadata: SkillMetadata
    path: Path
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class LoadedSkill(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    metadata: SkillMetadata
    body: str
    references: dict[str, str]
    path: Path
    content_hash: str

    def identity(self) -> dict[str, str]:
        return {
            "skill_name": self.metadata.name,
            "skill_version": self.metadata.version,
            "skill_content_hash": self.content_hash,
        }
