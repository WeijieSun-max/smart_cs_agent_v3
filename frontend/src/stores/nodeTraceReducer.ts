import type { NodeModelCall, NodeTrace, NodeTraceEvent } from "@/types/nodeTrace"

export interface NodeTraceState {
  nodeTraces: NodeTrace[]
  seenNodeTraceEvents: Record<string, true>
}

const eventKey = (event: NodeTraceEvent) => `${event.turn_id}:${event.sequence}`

function emptyTrace(event: NodeTraceEvent): NodeTrace {
  return {
    id: event.node_trace_id,
    turnId: event.turn_id,
    nodeName: event.node_name,
    status: "running",
    modelCalls: [],
    firstSequence: event.sequence,
    lastSequence: event.sequence,
    statusSequence: event.sequence,
  }
}

function emptyModelCall(event: NodeTraceEvent): NodeModelCall {
  return {
    id: event.model_call_id ?? `model-${event.sequence}`,
    status: "running",
    firstSequence: event.sequence,
    lastSequence: event.sequence,
    statusSequence: event.sequence,
  }
}

function updateModelCall(trace: NodeTrace, event: NodeTraceEvent): NodeTrace {
  const modelId = event.model_call_id ?? `model-${event.sequence}`
  const existing = trace.modelCalls.find((call) => call.id === modelId)
  const call = { ...(existing ?? emptyModelCall(event)), lastSequence: Math.max(existing?.lastSequence ?? event.sequence, event.sequence) }
  if (event.phase === "llm_start") {
    call.prompt = event.data.prompt
    call.startedAt = event.timestamp
    if (event.sequence >= call.statusSequence && !call.completedAt) {
      call.status = "running"
      call.statusSequence = event.sequence
    }
  } else if (event.sequence >= call.statusSequence) {
    call.response = event.data.response
    call.completedAt = event.timestamp
    call.status = "success"
    call.statusSequence = event.sequence
  }
  const modelCalls = existing
    ? trace.modelCalls.map((item) => item.id === modelId ? call : item)
    : [...trace.modelCalls, call]
  return { ...trace, modelCalls: modelCalls.sort((a, b) => a.firstSequence - b.firstSequence) }
}

export function reduceNodeTraceEvent(state: NodeTraceState, event: NodeTraceEvent): NodeTraceState {
  const key = eventKey(event)
  if (state.seenNodeTraceEvents[key]) return state

  const existing = state.nodeTraces.find((trace) => trace.id === event.node_trace_id)
  let trace: NodeTrace = {
    ...(existing ?? emptyTrace(event)),
    nodeName: event.node_name,
    firstSequence: Math.min(existing?.firstSequence ?? event.sequence, event.sequence),
    lastSequence: Math.max(existing?.lastSequence ?? event.sequence, event.sequence),
  }

  if (event.phase === "node_start") {
    trace.input = event.data.input
    trace.startedAt = event.timestamp
    if (event.sequence >= trace.statusSequence && !trace.completedAt) {
      trace.status = "running"
      trace.statusSequence = event.sequence
    }
  } else if (event.phase === "llm_start" || event.phase === "llm_end") {
    trace = updateModelCall(trace, event)
  } else if (event.phase === "node_end" && event.sequence >= trace.statusSequence) {
    trace.output = event.data.output
    trace.completedAt = event.timestamp
    trace.status = "success"
    trace.statusSequence = event.sequence
  } else if (event.phase === "node_error" && event.sequence >= trace.statusSequence) {
    trace.error = event.data.error
    trace.completedAt = event.timestamp
    trace.status = "error"
    trace.statusSequence = event.sequence
    trace.modelCalls = trace.modelCalls.map((call) => call.status === "running"
      ? { ...call, status: "error", completedAt: event.timestamp, statusSequence: event.sequence }
      : call)
  }

  const nodeTraces = existing
    ? state.nodeTraces.map((item) => item.id === trace.id ? trace : item)
    : [...state.nodeTraces, trace]
  return {
    nodeTraces: nodeTraces.sort((a, b) => a.firstSequence - b.firstSequence),
    seenNodeTraceEvents: { ...state.seenNodeTraceEvents, [key]: true },
  }
}
