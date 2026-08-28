from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class RunManifest(BaseModel):
    """Reproducibility metadata kept separate from task results and PII."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "smart-cs-eval-run/v1"
    run_id: str
    created_at: datetime
    git_sha: str | None = None
    dataset_path: str
    dataset_sha256: str
    trials: int = Field(ge=1)
    runner: str
    environment: str
    model_profiles: dict[str, str] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    skill_versions: dict[str, str] = Field(default_factory=dict)
    knowledge_versions: dict[str, str] = Field(default_factory=dict)
    migration_sha256: dict[str, str] = Field(default_factory=dict)
    random_seeds: tuple[int, ...] = ()
    python_version: str
    platform: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_run_manifest(
    dataset_path: Path,
    *,
    trials: int,
    runner: str,
    environment: str,
    repository_root: Path | None = None,
    model_profiles: dict[str, str] | None = None,
    prompt_versions: dict[str, str] | None = None,
    skill_versions: dict[str, str] | None = None,
    knowledge_versions: dict[str, str] | None = None,
    random_seeds: tuple[int, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> RunManifest:
    if trials < 1:
        raise ValueError("trials must be positive")
    resolved_dataset = dataset_path.resolve()
    repo = (repository_root or resolved_dataset.parent).resolve()
    migrations = repo / "migrations"
    return RunManifest(
        run_id=uuid4().hex,
        created_at=datetime.now(timezone.utc),
        git_sha=_git_sha(repo),
        dataset_path=str(resolved_dataset),
        dataset_sha256=file_sha256(resolved_dataset),
        trials=trials,
        runner=runner,
        environment=environment,
        model_profiles=dict(model_profiles or {}),
        prompt_versions=dict(prompt_versions or {}),
        skill_versions=dict(skill_versions or {}),
        knowledge_versions=dict(knowledge_versions or {}),
        migration_sha256={
            str(path.relative_to(repo)).replace("\\", "/"): file_sha256(path)
            for path in sorted(migrations.glob("*.sql"))
        } if migrations.is_dir() else {},
        random_seeds=random_seeds,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        metadata=dict(metadata or {}),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha(repository_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if len(value) == 40 else None
