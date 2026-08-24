from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KnowledgeRecord:
    point_id: str
    content: str
    payload: dict[str, Any]


def load_10086_qa(root_dir: Path) -> list[KnowledgeRecord]:
    records: list[KnowledgeRecord] = []
    for path in sorted((root_dir / "data" / "10086_qa").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = _first_heading(text) or path.stem
        source_url = _source_url(text)
        matches = list(re.finditer(r"(?m)^##\s+(\d+)\.\s+(.+?)\s*$", text))
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            answer = _clean_body(text[start:end])
            if not answer:
                continue
            question_index = int(match.group(1))
            question = match.group(2).strip()
            content = f"一级分类：{title}\n问题：{question}\n答案：{answer}"
            payload = {
                "knowledge_family": "10086",
                "source_type": "qa",
                "domain": "telecom",
                "document_type": "troubleshooting",
                "first_level_index": title,
                "question_index": question_index,
                "question": question,
                "answer": answer,
                "source": path.relative_to(root_dir).as_posix(),
                "source_file": path.name,
                "source_url": source_url,
                "content_sha256": _sha256(content),
                "status": "active",
                "version": "1.0.0",
                "language": "zh-CN",
            }
            records.append(KnowledgeRecord(_point_id("qa", payload), content, payload))
    return records


def load_10086_help(root_dir: Path) -> list[KnowledgeRecord]:
    records: list[KnowledgeRecord] = []
    for path in sorted((root_dir / "data" / "10086_help").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = _first_heading(text) or path.stem
        source_url = _source_url(text)
        image_urls = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
        body = _clean_body(text)
        if not body:
            continue
        content = f"标题：{title}\n正文：{body}"
        payload = {
            "knowledge_family": "10086",
            "source_type": "help",
            "domain": "retail",
            "document_type": "policy",
            "first_level_index": title,
            "title": title,
            "source": path.relative_to(root_dir).as_posix(),
            "source_file": path.name,
            "source_url": source_url,
            "image_urls": image_urls,
            "content_sha256": _sha256(content),
            "status": "active",
            "version": "1.0.0",
            "language": "zh-CN",
        }
        records.append(KnowledgeRecord(_point_id("help", payload), content, payload))
    return records


def load_10086_records(root_dir: Path) -> list[KnowledgeRecord]:
    return [*load_10086_qa(root_dir), *load_10086_help(root_dir)]


def _first_heading(text: str) -> str | None:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", text)
    return match.group(1).strip() if match else None


def _source_url(text: str) -> str:
    match = re.search(r"来源：\[[^\]]+\]\(([^)]+)\)", text)
    return match.group(1).strip() if match else ""


def _clean_body(text: str) -> str:
    text = re.sub(r"(?m)^# .+?$", "", text)
    text = re.sub(r"(?m)^---\s*$.*", "", text, flags=re.S)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _point_id(record_type: str, payload: dict[str, Any]) -> str:
    raw = "|".join(
        str(payload.get(key, ""))
        for key in ("knowledge_family", "source", "source_type", "question_index", "question", "title", "content_sha256")
    )
    return hashlib.sha256(f"{record_type}|{raw}".encode("utf-8")).hexdigest()[:32]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
