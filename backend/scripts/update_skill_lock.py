"""Publish newly versioned file Skills into the immutable release lock."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from domain.customer_service_agent.file_skills.integrity import update_skill_lock


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add new Skill versions to skills/skill-lock.json without rewriting released versions."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parents[1] / "skills",
        help="Skill root (default: backend/skills)",
    )
    args = parser.parse_args()
    entries = update_skill_lock(args.root)
    print(f"Skill release lock is current: {len(entries)} version(s)")


if __name__ == "__main__":
    main()
