from __future__ import annotations

from pathlib import Path

import pytest

from domain.customer_service_agent.file_skills.catalog import FileSkillCatalog
from domain.customer_service_agent.file_skills.integrity import load_skill_lock, update_skill_lock
from domain.customer_service_agent.tools.tool_registry import get_mcp_server


def _write_skill(root: Path, *, version: str, reference_text: str = "policy-v1") -> Path:
    skill_dir = root / "telecom" / "test-plan"
    references_dir = skill_dir / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    (references_dir / "policy.md").write_text(reference_text, encoding="utf-8")
    path = skill_dir / "SKILL.md"
    path.write_text(
        f"""---
name: test-plan-skill
description: A sufficiently long test description for immutable Skill release validation.
version: {version}
domain: telecom
capabilities:
  - test_plan_capability
allowed_agent_types:
  - telecom_agent
allowed_tools:
  - telecom_get_current_plan
effect: read
---

# Test Skill

Read [the policy](references/policy.md) before answering.
""",
        encoding="utf-8",
    )
    return path


def test_catalog_requires_a_release_locked_version(tmp_path: Path) -> None:
    _write_skill(tmp_path, version="1.0.0")

    with pytest.raises(ValueError, match="missing Skill release lock"):
        FileSkillCatalog(tmp_path, get_mcp_server()).scan_and_freeze()

    locked = update_skill_lock(tmp_path)
    catalog = FileSkillCatalog(tmp_path, get_mcp_server())
    catalog.scan_and_freeze()

    entry = catalog.select(capability="test_plan_capability", agent_type="telecom_agent")
    assert entry is not None
    assert locked["test-plan-skill@1.0.0"] == entry.content_hash


def test_released_version_cannot_change_instructions_or_references(tmp_path: Path) -> None:
    _write_skill(tmp_path, version="1.0.0")
    update_skill_lock(tmp_path)
    catalog = FileSkillCatalog(tmp_path, get_mcp_server())
    catalog.scan_and_freeze()

    _write_skill(tmp_path, version="1.0.0", reference_text="policy-was-modified")

    with pytest.raises(RuntimeError, match="changed after catalog freeze"):
        catalog.load("test-plan-skill", agent_type="telecom_agent")
    with pytest.raises(ValueError, match="refusing to overwrite released Skill"):
        update_skill_lock(tmp_path)
    with pytest.raises(ValueError, match="content changed without a version bump"):
        FileSkillCatalog(tmp_path, get_mcp_server()).scan_and_freeze()


def test_version_bump_adds_release_and_new_catalog_selects_it(tmp_path: Path) -> None:
    _write_skill(tmp_path, version="1.0.0")
    first_lock = update_skill_lock(tmp_path)

    _write_skill(tmp_path, version="1.0.1", reference_text="policy-v2")
    second_lock = update_skill_lock(tmp_path)
    catalog = FileSkillCatalog(tmp_path, get_mcp_server())
    catalog.scan_and_freeze()

    entry = catalog.select(capability="test_plan_capability", agent_type="telecom_agent")
    assert entry is not None
    assert entry.metadata.version == "1.0.1"
    assert second_lock["test-plan-skill@1.0.0"] == first_lock["test-plan-skill@1.0.0"]
    assert second_lock["test-plan-skill@1.0.1"] == entry.content_hash
    assert load_skill_lock(tmp_path) == second_lock


def test_new_version_must_move_forward_and_source_cannot_silently_roll_back(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, version="1.0.0")
    update_skill_lock(tmp_path)
    _write_skill(tmp_path, version="1.1.0")
    update_skill_lock(tmp_path)

    _write_skill(tmp_path, version="1.0.0")
    with pytest.raises(ValueError, match="rolled back below"):
        FileSkillCatalog(tmp_path, get_mcp_server()).scan_and_freeze()

    _write_skill(tmp_path, version="1.0.1")
    with pytest.raises(ValueError, match="must exceed the highest released version"):
        update_skill_lock(tmp_path)
