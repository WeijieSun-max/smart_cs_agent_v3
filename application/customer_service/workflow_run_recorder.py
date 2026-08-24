from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from application.customer_service import agent_run_service


@dataclass
class ActiveNode:
    step_id: str
    node_name: str
    started_at: str
    started_monotonic: float
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    usage_reported: bool = False


class WorkflowRunRecorder:
    def __init__(self, run, *, registry=None) -> None:
        self.run = run
        self.registry = registry or agent_run_service.registry
        self.active: dict[str, ActiveNode] = {}

    def start_node(self, event: dict[str, Any]) -> dict[str, Any] | None:
        node_name = graph_node_name(event)
        if event.get("event") != "on_chain_start" or node_name is None:
            return None
        run_id = str(event.get("run_id"))
        started_at = datetime.now(timezone.utc).isoformat()
        step = self.registry.start_step(
            self.run,
            node_default_title(node_name),
            node_name=node_name,
            step_type=step_type_for_node(node_name),
            started_at=started_at,
        )
        self.active[run_id] = ActiveNode(
            step_id=str(step["id"]),
            node_name=node_name,
            started_at=started_at,
            started_monotonic=perf_counter(),
        )
        return step

    def record_model_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("event")
        if event_type not in {"on_chat_model_start", "on_chat_model_end"}:
            return
        node_name = (event.get("metadata") or {}).get("langgraph_node")
        active = self._latest_active(str(node_name)) if node_name else None
        if active is None:
            return
        if event_type == "on_chat_model_start":
            active.model_calls += 1
            return
        if active.model_calls == 0:
            active.model_calls = 1
        usage = extract_token_usage((event.get("data") or {}).get("output"))
        if usage is not None:
            active.usage_reported = True
            active.prompt_tokens += usage["prompt"]
            active.completion_tokens += usage["completion"]
            active.total_tokens += usage["total"]

    def complete_node(self, event: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
        node_name = graph_node_name(event)
        if event.get("event") != "on_chain_end" or node_name is None:
            return None
        active = self.active.pop(str(event.get("run_id")), None)
        if active is None:
            return None
        output = (event.get("data") or {}).get("output")
        output = output if isinstance(output, dict) else {}
        completed_at = datetime.now(timezone.utc).isoformat()
        token_usage = None
        if active.usage_reported:
            token_usage = {
                "prompt": active.prompt_tokens,
                "completion": active.completion_tokens,
                "total": active.total_tokens,
            }
        step = self.registry.complete_step(
            self.run,
            active.step_id,
            title=node_output_title(node_name, output),
            completed_at=completed_at,
            duration_ms=round((perf_counter() - active.started_monotonic) * 1000),
            token_usage=token_usage,
            model_calls=active.model_calls,
        )
        return step, output

    def fail_active(self) -> None:
        for active in list(self.active.values()):
            token_usage = None
            if active.usage_reported:
                token_usage = {
                    "prompt": active.prompt_tokens,
                    "completion": active.completion_tokens,
                    "total": active.total_tokens,
                }
            self.registry.complete_step(
                self.run,
                active.step_id,
                status="error",
                duration_ms=round((perf_counter() - active.started_monotonic) * 1000),
                token_usage=token_usage,
                model_calls=active.model_calls,
            )
        self.active.clear()

    def _latest_active(self, node_name: str) -> ActiveNode | None:
        return next(
            (active for active in reversed(list(self.active.values())) if active.node_name == node_name),
            None,
        )


def graph_node_name(event: dict[str, Any]) -> str | None:
    metadata = event.get("metadata") or {}
    node_name = metadata.get("langgraph_node")
    if not node_name or event.get("name") != node_name:
        return None
    return str(node_name)


def node_default_title(node_name: str) -> str:
    return {
        "history_fusion_node": "融合会话历史上下文",
        "supervisor_node": "规划并执行 Telecom/Retail 任务",
        "compliance_checker_node": "执行合规审查",
        "response_synthesizer_node": "生成最终回复",
    }.get(node_name, node_name)


def node_output_title(node_name: str, output: dict[str, Any]) -> str:
    logs = output.get("node_logs") or []
    if isinstance(logs, str):
        logs = [logs]
    valid_logs = [str(log) for log in logs if log and log != "RESET"]
    return valid_logs[-1] if valid_logs else node_default_title(node_name)


def extract_token_usage(output: Any) -> dict[str, int] | None:
    candidates = [output]
    message = getattr(output, "message", None)
    if message is not None:
        candidates.append(message)
    generations = getattr(output, "generations", None)
    if generations:
        for generation_group in generations:
            group = generation_group if isinstance(generation_group, list) else [generation_group]
            for generation in group:
                candidates.extend([generation, getattr(generation, "message", None)])

    for candidate in candidates:
        if candidate is None:
            continue
        usage = getattr(candidate, "usage_metadata", None)
        response_metadata = getattr(candidate, "response_metadata", None) or {}
        if not usage and isinstance(candidate, dict):
            usage = candidate.get("usage_metadata")
            response_metadata = candidate.get("response_metadata") or response_metadata
        if not usage and isinstance(response_metadata, dict):
            usage = response_metadata.get("token_usage") or response_metadata.get("usage")
        normalized = normalize_token_usage(usage)
        if normalized is not None:
            return normalized
    return None


def normalize_token_usage(usage: Any) -> dict[str, int] | None:
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("input_tokens", usage.get("prompt_tokens"))
    completion = usage.get("output_tokens", usage.get("completion_tokens"))
    total = usage.get("total_tokens")
    if prompt is None and completion is None and total is None:
        return None
    prompt_value = int(prompt or 0)
    completion_value = int(completion or 0)
    return {
        "prompt": prompt_value,
        "completion": completion_value,
        "total": int(total if total is not None else prompt_value + completion_value),
    }


def step_type_for_node(node_name: str) -> str:
    if node_name == "history_fusion_node":
        return "thinking"
    if node_name == "supervisor_node":
        return "tool_call"
    if node_name == "response_synthesizer_node":
        return "final"
    return "thinking"
