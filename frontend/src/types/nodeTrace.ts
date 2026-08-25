export type NodeTracePhase = "node_start" | "llm_start" | "llm_end" | "node_end" | "node_error"
export type NodeTraceStatus = "running" | "success" | "error"

export interface NodeTraceEvent {
  type: "node_trace"
  phase: NodeTracePhase
  turn_id: string
  sequence: number
  node_trace_id: string
  node_name: string
  model_call_id?: string
  timestamp: string
  data: {
    input?: unknown
    prompt?: unknown
    response?: unknown
    output?: unknown
    error?: unknown
  }
}

export interface NodeModelCall {
  id: string
  status: NodeTraceStatus
  prompt?: unknown
  response?: unknown
  startedAt?: string
  completedAt?: string
  firstSequence: number
  lastSequence: number
  statusSequence: number
}

export interface NodeTrace {
  id: string
  turnId: string
  nodeName: string
  status: NodeTraceStatus
  input?: unknown
  output?: unknown
  error?: unknown
  modelCalls: NodeModelCall[]
  startedAt?: string
  completedAt?: string
  firstSequence: number
  lastSequence: number
  statusSequence: number
}
