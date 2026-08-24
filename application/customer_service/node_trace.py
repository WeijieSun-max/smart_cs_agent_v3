from __future__ import annotations

import json
import re
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

from langchain_core.messages import BaseMessage
from pydantic import BaseModel


MAX_DEBUG_PAYLOAD_BYTES = 1024 * 1024
MAX_DEBUG_DEPTH = 20
MAX_DEBUG_COLLECTION_ITEMS = 10_000

_CREDENTIAL_KEYS = {
    "apikey",
    "xapikey",
    "authorization",
    "proxyauthorization",
    "password",
    "passwd",
    "secret",
    "secretkey",
    "clientsecret",
    "accesstoken",
    "refreshtoken",
    "idtoken",
    "sessiontoken",
    "token",
    "cookie",
    "setcookie",
}
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_BASIC_AUTH_PATTERN = re.compile(r"(?i)\bBasic\s+[A-Za-z0-9+/=]{8,}")
_SECRET_KEY_PATTERN = re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{8,}")
_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(api[ _-]?key|authorization|password|passwd|secret(?:[ _-]?key)?|client[ _-]?secret|"
    r"access[ _-]?token|refresh[ _-]?token|cookie|set[ _-]?cookie)\s*([:=])\s*([^\s,;]+)"
)
_URL_CREDENTIAL_PATTERN = re.compile(r"(https?://[^:/\s]+:)[^@\s]+@", re.IGNORECASE)
_MEMORY_REFERENCE_PATTERN = re.compile(
    r"<<<MEMORY_REFERENCE_DATA>>>.*?<<<END_MEMORY_REFERENCE_DATA>>>",
    re.DOTALL,
)


def serialize_debug_value(value: Any, *, max_bytes: int = MAX_DEBUG_PAYLOAD_BYTES) -> Any:
    """Convert a runtime value into bounded, credential-safe JSON data."""
    try:
        converted = _convert_value(value, seen=set(), depth=0)
        redacted = _redact_credentials(converted)
        encoded = json.dumps(redacted, ensure_ascii=False, default=str)
    except Exception:
        return {"_debug_error": "UNSERIALIZABLE", "type": type(value).__name__}

    encoded_bytes = encoded.encode("utf-8")
    if len(encoded_bytes) <= max_bytes:
        return redacted
    preview = encoded_bytes[:max_bytes].decode("utf-8", errors="ignore")
    return {
        "_debug_truncated": True,
        "reason": f"payload exceeded {max_bytes} bytes",
        "original_bytes": len(encoded_bytes),
        "preview": preview,
    }


def _convert_value(value: Any, *, seen: set[int], depth: int) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if depth >= MAX_DEBUG_DEPTH:
        return {"_debug_truncated": True, "reason": "maximum depth exceeded"}
    if isinstance(value, Enum):
        return _convert_value(value.value, seen=seen, depth=depth + 1)
    if isinstance(value, (datetime, date, UUID, Path)):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, BaseMessage):
        message: dict[str, Any] = {
            "type": value.type,
            "content": value.content,
            "additional_kwargs": value.additional_kwargs,
            "response_metadata": value.response_metadata,
        }
        for attribute in ("name", "id", "usage_metadata"):
            attribute_value = getattr(value, attribute, None)
            if attribute_value is not None:
                message[attribute] = attribute_value
        return _convert_value(message, seen=seen, depth=depth + 1)

    value_id = id(value)
    if value_id in seen:
        return {"_debug_error": "CIRCULAR_REFERENCE", "type": type(value).__name__}

    if isinstance(value, BaseModel):
        seen.add(value_id)
        try:
            return _convert_value(value.model_dump(), seen=seen, depth=depth + 1)
        finally:
            seen.discard(value_id)
    if is_dataclass(value) and not isinstance(value, type):
        seen.add(value_id)
        try:
            mapped = {field.name: getattr(value, field.name) for field in fields(value)}
            return _convert_value(mapped, seen=seen, depth=depth + 1)
        finally:
            seen.discard(value_id)
    if isinstance(value, Mapping):
        seen.add(value_id)
        try:
            items = list(value.items())
            mapped: dict[str, Any] = {}
            for key, item in items[:MAX_DEBUG_COLLECTION_ITEMS]:
                mapped[str(key)] = _convert_value(
                    item, seen=seen, depth=depth + 1
                )
            if len(items) > MAX_DEBUG_COLLECTION_ITEMS:
                mapped["_debug_truncated"] = {
                    "reason": "collection item limit exceeded",
                    "original_items": len(items),
                }
            return mapped
        finally:
            seen.discard(value_id)
    if isinstance(value, (list, tuple, set, frozenset)):
        seen.add(value_id)
        try:
            items = list(value)
            converted = [
                _convert_value(item, seen=seen, depth=depth + 1)
                for item in items[:MAX_DEBUG_COLLECTION_ITEMS]
            ]
            if len(items) > MAX_DEBUG_COLLECTION_ITEMS:
                converted.append({
                    "_debug_truncated": True,
                    "reason": "collection item limit exceeded",
                    "original_items": len(items),
                })
            return converted
        finally:
            seen.discard(value_id)

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _convert_value(tolist(), seen=seen, depth=depth + 1)
        except Exception:
            pass
    return {"_debug_type": type(value).__name__, "repr": _safe_repr(value)}


def _redact_credentials(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, list):
        return [_redact_credentials(item) for item in value]
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = re.sub(r"[^a-z0-9]", "", str(key).strip().lower())
            if normalized_key == "memorypacket" and isinstance(item, dict):
                redacted[str(key)] = _memory_packet_trace_summary(item)
            else:
                redacted[str(key)] = (
                    "[REDACTED_CREDENTIAL]"
                    if _is_credential_key(normalized_key)
                    else _redact_credentials(item)
                )
        return redacted
    return value


def _is_credential_key(normalized_key: str) -> bool:
    if normalized_key in _CREDENTIAL_KEYS:
        return True
    return normalized_key.endswith((
        "apikey",
        "password",
        "passwd",
        "secret",
        "secretkey",
        "accesstoken",
        "refreshtoken",
        "idtoken",
        "sessiontoken",
    ))


def _redact_string(value: str) -> str:
    redacted = _MEMORY_REFERENCE_PATTERN.sub("[REDACTED_MEMORY_REFERENCE_DATA]", value)
    redacted = _BEARER_PATTERN.sub("Bearer [REDACTED_CREDENTIAL]", redacted)
    redacted = _BASIC_AUTH_PATTERN.sub("Basic [REDACTED_CREDENTIAL]", redacted)
    redacted = _SECRET_KEY_PATTERN.sub("[REDACTED_CREDENTIAL]", redacted)
    redacted = _ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED_CREDENTIAL]",
        redacted,
    )
    return _URL_CREDENTIAL_PATTERN.sub(r"\1[REDACTED_CREDENTIAL]@", redacted)


def _memory_packet_trace_summary(packet: dict[str, Any]) -> dict[str, Any]:
    def _count(key: str) -> int:
        value = packet.get(key)
        return len(value) if isinstance(value, list) else 0

    memory_types = sorted({
        str(item.get("memory_type"))
        for key in ("episodes", "semantic_memories")
        for item in (packet.get(key) or [])
        if isinstance(item, dict) and item.get("memory_type")
    })
    return {
        "session_summary_present": bool(packet.get("session_summary")),
        "recent_message_count": _count("recent_messages"),
        "episode_count": _count("episodes"),
        "semantic_memory_count": _count("semantic_memories"),
        "memory_types": memory_types,
        "token_count": packet.get("token_count", 0),
        "max_tokens": packet.get("max_tokens", 0),
    }


def _safe_repr(value: Any) -> str:
    try:
        return _redact_string(repr(value))
    except Exception:
        return f"<{type(value).__name__}>"


class NodeTraceRecorder:
    """Map one request's LangGraph events to live-only Node Trace patches."""

    def __init__(self, turn_id: str) -> None:
        self.turn_id = turn_id
        self._sequence = 0
        self._active_nodes: dict[str, str] = {}
        self._model_owners: dict[str, str] = {}

    def consume(self, event: dict[str, Any]) -> dict[str, Any] | None:
        event_type = str(event.get("event") or "")
        if event_type == "on_chain_start":
            return self._start_node(event)
        if event_type == "on_chat_model_start":
            return self._start_model(event)
        if event_type == "on_chat_model_end":
            return self._end_model(event)
        if event_type == "on_chain_end":
            return self._end_node(event)
        return None

    def fail_active(self, error: BaseException) -> list[dict[str, Any]]:
        error_payload = serialize_debug_value({
            "type": type(error).__name__,
            "message": str(error),
        })
        events = [
            self._event(
                "node_error",
                node_trace_id=node_trace_id,
                node_name=node_name,
                data={"error": error_payload},
            )
            for node_trace_id, node_name in list(self._active_nodes.items())
        ]
        self._active_nodes.clear()
        self._model_owners.clear()
        return events

    def _start_node(self, event: dict[str, Any]) -> dict[str, Any] | None:
        node_name = _graph_node_name(event)
        if node_name is None:
            return None
        node_trace_id = str(event.get("run_id"))
        self._active_nodes[node_trace_id] = node_name
        raw_input = (event.get("data") or {}).get("input")
        return self._event(
            "node_start",
            node_trace_id=node_trace_id,
            node_name=node_name,
            data={"input": serialize_debug_value(raw_input)},
        )

    def _start_model(self, event: dict[str, Any]) -> dict[str, Any] | None:
        owner = self._resolve_node(event)
        if owner is None:
            return None
        node_trace_id, node_name = owner
        model_call_id = str(event.get("run_id"))
        self._model_owners[model_call_id] = node_trace_id
        raw_prompt = (event.get("data") or {}).get("input")
        prompt = serialize_debug_value(raw_prompt)
        return self._event(
            "llm_start",
            node_trace_id=node_trace_id,
            node_name=node_name,
            model_call_id=model_call_id,
            data={"prompt": prompt},
        )

    def _end_model(self, event: dict[str, Any]) -> dict[str, Any] | None:
        model_call_id = str(event.get("run_id"))
        node_trace_id = self._model_owners.pop(model_call_id, None)
        if node_trace_id is None:
            owner = self._resolve_node(event)
            if owner is None:
                return None
            node_trace_id, node_name = owner
        else:
            node_name = self._active_nodes.get(node_trace_id)
            if node_name is None:
                return None
        raw_response = (event.get("data") or {}).get("output")
        response = serialize_debug_value(raw_response)
        return self._event(
            "llm_end",
            node_trace_id=node_trace_id,
            node_name=node_name,
            model_call_id=model_call_id,
            data={"response": response},
        )

    def _end_node(self, event: dict[str, Any]) -> dict[str, Any] | None:
        node_trace_id = str(event.get("run_id"))
        node_name = self._active_nodes.pop(node_trace_id, None)
        if node_name is None or _graph_node_name(event) is None:
            return None
        stale_model_calls = [
            call_id for call_id, owner_id in self._model_owners.items() if owner_id == node_trace_id
        ]
        for call_id in stale_model_calls:
            self._model_owners.pop(call_id, None)
        raw_output = (event.get("data") or {}).get("output")
        output = serialize_debug_value(raw_output)
        return self._event(
            "node_end",
            node_trace_id=node_trace_id,
            node_name=node_name,
            data={"output": output},
        )

    def _resolve_node(self, event: dict[str, Any]) -> tuple[str, str] | None:
        for parent_id in reversed(event.get("parent_ids") or []):
            node_trace_id = str(parent_id)
            node_name = self._active_nodes.get(node_trace_id)
            if node_name is not None:
                return node_trace_id, node_name
        metadata_node = str((event.get("metadata") or {}).get("langgraph_node") or "")
        if metadata_node:
            for node_trace_id, node_name in reversed(list(self._active_nodes.items())):
                if node_name == metadata_node:
                    return node_trace_id, node_name
        return None

    def _event(
        self,
        phase: str,
        *,
        node_trace_id: str,
        node_name: str,
        data: dict[str, Any],
        model_call_id: str | None = None,
    ) -> dict[str, Any]:
        self._sequence += 1
        payload: dict[str, Any] = {
            "type": "node_trace",
            "phase": phase,
            "turn_id": self.turn_id,
            "sequence": self._sequence,
            "node_trace_id": node_trace_id,
            "node_name": node_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        if model_call_id is not None:
            payload["model_call_id"] = model_call_id
        return payload


def _graph_node_name(event: dict[str, Any]) -> str | None:
    metadata = event.get("metadata") or {}
    node_name = metadata.get("langgraph_node")
    if not node_name or event.get("name") != node_name:
        return None
    return str(node_name)
