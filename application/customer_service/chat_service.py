from __future__ import annotations

import asyncio
from typing import Any

from fastapi.responses import StreamingResponse

from adapter.web.schemas.chat import ChatRequest, ChatResponse, ChatStreamRequest
from application.customer_service import agent_run_service
from application.customer_service.admission import turn_admission
from application.customer_service.node_trace import NodeTraceRecorder
from application.customer_service.stream_events import (
    encode_sse as _encode_sse,
    generate_replay_events as _generate_replay_events,
    response_delta_chunks as _response_delta_chunks,
)
from application.customer_service.turn_context import (
    clear_checkpoint as _clear_checkpoint,
    persist_assistant_turn as _persist_assistant_turn,
    persist_pending_task as _persist_pending_task,
    prepare_turn as _prepare_turn,
    validate_turn_request as _validate_turn_request,
)
from application.customer_service.turn_lease_service import TurnLease, get_turn_lease_manager
from application.customer_service.workflow_run_recorder import WorkflowRunRecorder as _WorkflowRunRecorder
from domain.customer_service_agent.service import short_term_memory_service
from domain.customer_service_agent.workflow import customer_service_workflow
from domain.shared.checkpoint import checkpoint_saver_service
from pkg.config.settings import Settings, get_settings
from pkg.log.logger import get_logger
from pkg.telemetry import (
    finalize_turn,
    get_callback_handler,
    normalize_error,
    trace_turn,
)

logger = get_logger()
SAFE_ERROR_MESSAGE = "系统处理异常，请稍后重试。"


async def chat(request: ChatRequest) -> ChatResponse:
    async with turn_admission.slot(request.user_id or "anonymous", request.session_id):
        return await _chat_admitted(request)


async def _chat_admitted(request: ChatRequest) -> ChatResponse:
    graph = customer_service_workflow.get_workflow()
    user_id, state, turn, replay = await asyncio.to_thread(_prepare_turn, request)
    session_id = state["session_id"]
    if replay is not None:
        return ChatResponse(
            response=replay["content"],
            session_id=session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            intent="replayed",
            compliance_passed=True,
        )
    async with get_turn_lease_manager().hold(user_id, session_id, turn.turn_id) as lease:
        return await _execute_chat_turn(request, graph, user_id, state, turn, lease)


async def _execute_chat_turn(request, graph, user_id: str, state: dict[str, Any], turn, lease: TurnLease) -> ChatResponse:
    settings = get_settings()
    model, provider = _resolve_model_identity(settings)
    session_id = state["session_id"]
    run = await asyncio.to_thread(
        agent_run_service.registry.begin,
        session_id,
        turn.turn_id,
        user_id=user_id,
    )
    run.stop_requested = lease.stop_requested
    run.attach_current_task()
    memory = short_term_memory_service.get_service()
    try:
        if not state.get("user_message_persisted"):
            await asyncio.to_thread(
                memory.add_message,
                session_id,
                "user",
                request.message,
                turn.turn_id,
                user_id=user_id,
            )
        await asyncio.to_thread(
            agent_run_service.registry.record_step,
            run,
            "接收用户请求",
            node_name="user_query",
            step_type="query",
        )
    except Exception:
        agent_run_service.registry.finish(run, "failed")
        raise

    result: dict[str, Any] = {}
    final_response = ""
    caught: tuple[Exception, Any] | None = None
    recorder = _WorkflowRunRecorder(run)

    with trace_turn(
        turn,
        model=model,
        provider=provider,
        input_text=request.message,
    ) as root:
        try:
            config = _build_graph_config(
                user_id,
                session_id,
                turn.turn_id,
                checkpoint_ns=lease.checkpoint_namespace,
            )
            async for event in graph.astream_events(state, config=config, version="v2"):
                recorder.record_model_event(event)
                recorder.start_node(event)
                completed = recorder.complete_node(event)
                if completed is not None:
                    _, output = completed
                    result.update(output)
            final_response = result.get("final_response") or SAFE_ERROR_MESSAGE
            await asyncio.to_thread(
                _persist_pending_task,
                memory,
                session_id,
                result.get("pending_task", state.get("pending_task")),
                user_id=user_id,
            )
            await asyncio.to_thread(
                _persist_assistant_turn,
                memory,
                session_id,
                final_response,
                turn.turn_id,
                settings.memory_layered_enabled,
                user_id,
                lease,
            )
        except asyncio.CancelledError as exc:
            recorder.fail_active()
            finalize_turn(
                root,
                workflow_completed=False,
                response_delivered=False,
                error=None if run.stop_requested.is_set() else exc,
            )
            agent_run_service.registry.finish(run, "stopped" if run.stop_requested.is_set() else "cancelled")
            _clear_checkpoint(user_id, session_id, lease.checkpoint_namespace)
            raise
        except Exception as exc:
            recorder.fail_active()
            caught = (exc, exc.__traceback__)
            finalize_turn(
                root,
                workflow_completed=False,
                response_delivered=False,
                error=exc,
            )
        else:
            finalize_turn(
                root,
                workflow_completed=True,
                response_delivered=True,
                final_response=final_response,
                compliance_passed=result.get("compliance_passed", True),
            )

    if caught is not None:
        agent_run_service.registry.finish(run, "failed")
        _clear_checkpoint(user_id, session_id, lease.checkpoint_namespace)
        exc, traceback = caught
        raise exc.with_traceback(traceback)
    agent_run_service.registry.finish(run, "completed")
    _clear_checkpoint(user_id, session_id, lease.checkpoint_namespace)
    return ChatResponse(
        response=final_response,
        session_id=session_id,
        turn_id=turn.turn_id,
        trace_id=turn.trace_id,
        intent=result.get("intent") or "unknown",
        compliance_passed=result.get("compliance_passed", True),
    )


def chat_stream(request: ChatStreamRequest) -> StreamingResponse:
    _validate_turn_request(request)
    return StreamingResponse(
        _generate_stream_events(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def _build_graph_config(
    user_id: str | None,
    session_id: str,
    turn_id: str | None = None,
    *,
    checkpoint_ns: str = "",
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "configurable": {
            "thread_id": checkpoint_saver_service.instance.get_thread_id(user_id, session_id),
            "checkpoint_ns": checkpoint_ns,
        },
        "run_name": "customer-service-agent",
        "metadata": {"turn_id": turn_id or "untracked"},
    }
    handler = get_callback_handler()
    if handler is not None:
        config["callbacks"] = [handler]
    return config


async def _generate_stream_events(request: ChatStreamRequest):
    async with turn_admission.slot(request.user_id or "anonymous", request.session_id):
        graph = customer_service_workflow.get_workflow()
        user_id, state, turn, replay = await asyncio.to_thread(_prepare_turn, request)
        session_id = state["session_id"]
        if replay is not None:
            async for item in _generate_replay_events(turn, replay["content"]):
                yield item
            return
        async with get_turn_lease_manager().hold(user_id, session_id, turn.turn_id) as lease:
            run = await asyncio.to_thread(
                agent_run_service.registry.begin,
                session_id,
                turn.turn_id,
                user_id=user_id,
            )
            run.stop_requested = lease.stop_requested
            try:
                if not state.get("user_message_persisted"):
                    await asyncio.to_thread(
                        short_term_memory_service.get_service().add_message,
                        session_id,
                        "user",
                        request.message,
                        turn.turn_id,
                        user_id=user_id,
                    )
                await asyncio.to_thread(
                    agent_run_service.registry.record_step,
                    run,
                    "接收用户请求",
                    node_name="user_query",
                    step_type="query",
                )
            except Exception:
                agent_run_service.registry.finish(run, "failed")
                raise
            async for item in _generate_stream_events_admitted(
                graph,
                state,
                session_id,
                user_id,
                turn,
                run,
                lease=lease,
            ):
                yield item


async def _generate_stream_events_admitted(
    graph,
    chat_state,
    session_id: str,
    user_id: str | None,
    turn,
    run=None,
    *,
    lease: TurnLease | None = None,
):
    run = run or agent_run_service.registry.begin(session_id, turn.turn_id, user_id=user_id)
    run.attach_current_task()
    recorder = _WorkflowRunRecorder(run)
    trace_recorder = NodeTraceRecorder(turn.turn_id)
    final_response = ""
    pending_task = chat_state.get("pending_task")
    compliance_passed = True
    settings = get_settings()
    model, provider = _resolve_model_identity(settings)
    error_payload: dict[str, str] | None = None
    control_flow_error: BaseException | None = None
    stopped = False
    trace_error_events: list[dict[str, Any]] = []

    with trace_turn(
        turn,
        model=model,
        provider=provider,
        input_text=chat_state.get("raw_query"),
    ) as root:
        yield _encode_sse({"type": "meta", "turn_id": turn.turn_id, "trace_id": turn.trace_id})
        if run.steps:
            yield _encode_sse({"type": "step_complete", "step": run.steps[0]})
        try:
            config = _build_graph_config(
                user_id,
                session_id,
                turn.turn_id,
                checkpoint_ns=lease.checkpoint_namespace if lease is not None else "",
            )
            async for event in graph.astream_events(chat_state, config=config, version="v2"):
                try:
                    trace_event = trace_recorder.consume(event)
                except Exception as trace_error:
                    logger.warning(
                        "Node Trace event conversion failed event_type={} error_type={} turn_id={}",
                        event.get("event"),
                        type(trace_error).__name__,
                        turn.turn_id,
                    )
                else:
                    if trace_event is not None:
                        yield _encode_sse(trace_event)
                recorder.record_model_event(event)
                started_step = recorder.start_node(event)
                if started_step is not None:
                    yield _encode_sse({"type": "step_start", "step": started_step})
                    continue
                completed = recorder.complete_node(event)
                if completed is None:
                    continue
                step, output = completed
                yield _encode_sse({"type": "step_complete", "step": step})
                if "compliance_passed" in output:
                    compliance_passed = bool(output.get("compliance_passed", True))
                if "pending_task" in output:
                    pending_task = output.get("pending_task")
                if "final_response" in output:
                    final_response = output.get("final_response") or ""
                    for delta in _response_delta_chunks(final_response):
                        yield _encode_sse({"type": "message_delta", "delta": delta})
                        await asyncio.sleep(0)
                    yield _encode_sse({"type": "answer", "content": final_response})

            if final_response:
                await asyncio.to_thread(
                    _persist_pending_task,
                    short_term_memory_service.get_service(),
                    session_id,
                    pending_task,
                    user_id=user_id or "anonymous",
                )
                await asyncio.to_thread(
                    _persist_assistant_turn,
                    short_term_memory_service.get_service(),
                    session_id,
                    final_response,
                    turn.turn_id,
                    settings.memory_layered_enabled,
                    user_id,
                    lease,
                )
            finalize_turn(
                root,
                workflow_completed=True,
                response_delivered=bool(final_response),
                final_response=final_response,
                compliance_passed=compliance_passed,
            )
        except asyncio.CancelledError as exc:
            recorder.fail_active()
            stopped = run.stop_requested.is_set()
            if stopped:
                trace_error_events = trace_recorder.fail_active(RuntimeError("Agent execution stopped by user"))
            if not stopped:
                control_flow_error = exc
            finalize_turn(
                root,
                workflow_completed=False,
                response_delivered=False,
                error=None if stopped else exc,
            )
        except GeneratorExit as exc:
            recorder.fail_active()
            trace_recorder.fail_active(exc)
            control_flow_error = exc
            finalize_turn(
                root,
                workflow_completed=False,
                response_delivered=False,
                error=exc,
            )
        except Exception as exc:
            recorder.fail_active()
            trace_error_events = trace_recorder.fail_active(exc)
            safe_error = normalize_error(exc)
            logger.error(
                "Stream execution failed error_type={} error_code={} turn_id={}",
                safe_error["error_type"],
                safe_error["error_code"],
                turn.turn_id,
            )
            error_payload = {"type": "error", "content": SAFE_ERROR_MESSAGE, "turn_id": turn.turn_id}
            finalize_turn(
                root,
                workflow_completed=False,
                response_delivered=False,
                error=exc,
            )

    if control_flow_error is not None:
        agent_run_service.registry.finish(run, "cancelled")
        _clear_checkpoint(
            user_id,
            session_id,
            lease.checkpoint_namespace if lease is not None else None,
        )
        raise control_flow_error
    for trace_error_event in trace_error_events:
        yield _encode_sse(trace_error_event)
    if stopped:
        agent_run_service.registry.finish(run, "stopped")
        _clear_checkpoint(
            user_id,
            session_id,
            lease.checkpoint_namespace if lease is not None else None,
        )
        yield _encode_sse({"type": "stopped", "content": "Agent 已停止", "turn_id": turn.turn_id})
        yield _encode_sse({"type": "terminal", "status": "stopped", "turn_id": turn.turn_id})
        yield "data:[DONE]\n\n"
        return
    if error_payload is not None:
        agent_run_service.registry.finish(run, "failed")
        _clear_checkpoint(
            user_id,
            session_id,
            lease.checkpoint_namespace if lease is not None else None,
        )
        yield _encode_sse(error_payload)
        yield _encode_sse({"type": "terminal", "status": "failed", "turn_id": turn.turn_id})
    else:
        agent_run_service.registry.finish(run, "completed")
        _clear_checkpoint(
            user_id,
            session_id,
            lease.checkpoint_namespace if lease is not None else None,
        )
        yield _encode_sse({"type": "terminal", "status": "completed", "turn_id": turn.turn_id})
    yield "data:[DONE]\n\n"


def _resolve_model_identity(settings: Settings) -> tuple[str, str]:
    return settings.qwen_model, "openai-compatible"
