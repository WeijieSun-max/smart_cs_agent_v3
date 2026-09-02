"""Skill bundle hashing and immutable release-lock management."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import SkillMetadata

LOCK_FILE_NAME = "skill-lock.json"
LOCK_SCHEMA_VERSION = 1

_FRONTMATTER = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---\s*\r?\n", re.DOTALL)
_REFERENCE_LINK = re.compile(r"\[[^\]]+\]\((references/[^)#?]+)(?:#[^)]+)?\)")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY = re.compile(
    r"^(?P<name>[a-z][a-z0-9-]{2,63})@"
    r"(?P<version>(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))$"
)


@dataclass(frozen=True)
class SkillSource:
    """Validated on-disk content used to build or verify a catalog entry."""

    path: Path
    metadata: SkillMetadata
    body: str
    references: dict[str, str]
    content_hash: str

    @property
    def identity(self) -> str:
        return f"{self.metadata.name}@{self.metadata.version}"


def discover_skill_sources(root: Path) -> list[SkillSource]:
    """Read every Skill bundle below ``root`` in deterministic path order."""

    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"skill root does not exist: {resolved_root}")
    paths = sorted(resolved_root.rglob("SKILL.md"))
    if not paths:
        raise ValueError("no SKILL.md files found")
    return [read_skill_source(path) for path in paths]


def read_skill_source(path: Path) -> SkillSource:
    """Read one Skill and hash its instructions plus all explicit references."""

    resolved_path = path.resolve()
    raw = resolved_path.read_text(encoding="utf-8")
    match = _FRONTMATTER.match(raw)
    if match is None:
        raise ValueError(f"missing YAML frontmatter: {resolved_path}")
    parsed = yaml.safe_load(match.group(1))
    if not isinstance(parsed, dict):
        raise ValueError(f"invalid YAML frontmatter: {resolved_path}")
    metadata = SkillMetadata.model_validate(parsed)
    body = raw[match.end():].strip()
    references: dict[str, str] = {}
    skill_dir = resolved_path.parent.resolve()
    for relative in sorted(set(_REFERENCE_LINK.findall(body))):
        target = (skill_dir / relative).resolve()
        if skill_dir not in target.parents or not target.is_file():
            raise ValueError(f"invalid skill reference: {relative}")
        references[relative] = target.read_text(encoding="utf-8")
    return SkillSource(
        path=resolved_path,
        metadata=metadata,
        body=body,
        references=references,
        content_hash=_bundle_hash(raw, references),
    )


def load_skill_lock(root: Path, *, required: bool = True) -> dict[str, str]:
    """Load and strictly validate the committed immutable Skill release lock."""

    path = root.resolve() / LOCK_FILE_NAME
    if not path.is_file():
        if required:
            raise ValueError(
                f"missing Skill release lock: {path}; run python scripts/update_skill_lock.py"
            )
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Skill release lock: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise ValueError(f"unsupported Skill release lock schema: {path}")
    skills = payload.get("skills")
    if not isinstance(skills, dict):
        raise ValueError(f"invalid Skill release lock entries: {path}")
    result: dict[str, str] = {}
    for identity, content_hash in skills.items():
        if (
            not isinstance(identity, str)
            or _IDENTITY.fullmatch(identity) is None
            or not isinstance(content_hash, str)
            or _HASH.fullmatch(content_hash) is None
        ):
            raise ValueError(f"invalid Skill release lock entry: {identity!r}")
        result[identity] = content_hash
    return result


def validate_skill_lock(root: Path, sources: list[SkillSource]) -> None:
    """Require every source version to match its immutable release hash."""

    locked = load_skill_lock(root)
    for source in sources:
        expected = locked.get(source.identity)
        if expected is None:
            raise ValueError(
                f"Skill version is not release-locked: {source.identity}; "
                "run python scripts/update_skill_lock.py"
            )
        if expected != source.content_hash:
            raise ValueError(
                f"Skill content changed without a version bump: {source.identity}; "
                "restore the released content or increment the frontmatter version"
            )
    current_versions = _versions_by_name(source.identity for source in sources)
    locked_versions = _versions_by_name(locked)
    for name, versions in current_versions.items():
        if max(versions) < max(locked_versions[name]):
            raise ValueError(
                f"Skill source rolled back below its highest released version: {name}; "
                "publish a new higher version instead"
            )


def update_skill_lock(root: Path) -> dict[str, str]:
    """Add new versions to the lock while refusing mutation of released versions."""

    resolved_root = root.resolve()
    sources = discover_skill_sources(resolved_root)
    current: dict[str, str] = {}
    for source in sources:
        if source.identity in current:
            raise ValueError(f"duplicate skill name/version: {source.identity}")
        current[source.identity] = source.content_hash

    locked = load_skill_lock(resolved_root, required=False)
    locked_versions = _versions_by_name(locked)
    for identity, content_hash in current.items():
        previous = locked.get(identity)
        if previous is not None and previous != content_hash:
            raise ValueError(
                f"refusing to overwrite released Skill: {identity}; "
                "increment the frontmatter version first"
            )
        if previous is None:
            match = _IDENTITY.fullmatch(identity)
            assert match is not None
            name = match.group("name")
            version = _semver(match.group("version"))
            if name in locked_versions and version <= max(locked_versions[name]):
                raise ValueError(
                    f"new Skill version must exceed the highest released version: {identity}"
                )

    merged = {**locked, **current}
    payload = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "skills": dict(sorted(merged.items())),
    }
    destination = resolved_root / LOCK_FILE_NAME
    handle, temporary_name = tempfile.mkstemp(
        dir=resolved_root,
        prefix=f".{LOCK_FILE_NAME}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return merged


def _bundle_hash(raw: str, references: dict[str, str]) -> str:
    """Hash framed path/content pairs so reference edits change Skill identity."""

    digest = hashlib.sha256()
    _hash_part(digest, "SKILL.md", raw)
    for relative, content in sorted(references.items()):
        _hash_part(digest, relative, content)
    return digest.hexdigest()


def _hash_part(digest: Any, relative: str, content: str) -> None:
    path_bytes = relative.encode("utf-8")
    content_bytes = content.encode("utf-8")
    digest.update(len(path_bytes).to_bytes(8, "big"))
    digest.update(path_bytes)
    digest.update(len(content_bytes).to_bytes(8, "big"))
    digest.update(content_bytes)


def _versions_by_name(identities: Iterable[str]) -> dict[str, list[tuple[int, int, int]]]:
    result: dict[str, list[tuple[int, int, int]]] = {}
    for identity in identities:
        match = _IDENTITY.fullmatch(identity)
        assert match is not None
        result.setdefault(match.group("name"), []).append(_semver(match.group("version")))
    return result


def _semver(value: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in value.split("."))  # type: ignore[return-value]
