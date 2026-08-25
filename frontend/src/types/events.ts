import type { AgentStep } from "./agent"
import type { NodeTraceEvent, NodeTracePhase } from "./nodeTrace"

export type AgentEventType = "agent_start" | "planning" | "step_start" | "tool_call" | "tool_result" | "message_delta" | "step_complete" | "agent_complete" | "agent_error" | "meta" | "log" | "answer" | "error" | "stopped" | "terminal" | "node_trace" | "skill_event"
export type AgentTerminalStatus = "completed" | "failed" | "stopped" | "cancelled"
export type SkillEventName = "skill_candidates" | "skill_selected" | "skill_llm_decision" | "skill_tool_start" | "skill_tool_end" | "skill_confirmation_required" | "skill_completed"
export type SkillEventStatus = "candidates" | "selected" | "started" | "success" | "failed" | "rejected" | "call_tool" | "ask_clarification" | "request_confirmation" | "finish" | "needs_clarification" | "awaiting_confirmation" | "indeterminate" | "budget_exhausted" | "completed"
const NODE_TRACE_PHASES: readonly string[] = ["node_start", "llm_start", "llm_end", "node_end", "node_error"]

export interface SkillAgentEvent {
  type: "skill_event"
  event: SkillEventName
  status?: SkillEventStatus
  skill_name?: "onboarding_process_guide" | "onboarding_material_check" | "onboarding_risk_assessment" | "onboarding_eligibility_check" | "none"
  tool_name?: "knowledge_search" | "user_profile" | "risk_check" | "ticket_create"
  duration_ms?: number
  confidence?: number
  error_code?: "skill.invalid_decision" | "skill.tool_failed" | "skill.write_status_unknown" | "skill.tool_not_allowed" | "skill.validation_failed" | "skill.unauthorized_write" | "tool.validation"
  candidate_count?: number
  selector_llm_decisions?: number
  runtime_llm_decisions?: number
  remaining_llm_decisions?: number
  tool_calls?: number
  remaining_tool_calls?: number
}

export interface StandardAgentEvent {
  type: Exclude<AgentEventType, "skill_event">
  content?: string
  delta?: string
  step?: AgentStep
  stepId?: string
  sessionId?: string
  turn_id?: string
  trace_id?: string
  status?: AgentTerminalStatus
  metadata?: Record<string, unknown>
  phase?: NodeTracePhase
  sequence?: number
  node_trace_id?: string
  node_name?: string
  model_call_id?: string
  timestamp?: string
  data?: NodeTraceEvent["data"]
}

export type AgentEvent = StandardAgentEvent | SkillAgentEvent

export function isNodeTraceEvent(event: AgentEvent): event is StandardAgentEvent & NodeTraceEvent {
  return event.type === "node_trace"
    && typeof event.phase === "string"
    && NODE_TRACE_PHASES.includes(event.phase)
    && typeof event.turn_id === "string"
    && typeof event.sequence === "number"
    && typeof event.node_trace_id === "string"
    && typeof event.node_name === "string"
    && typeof event.timestamp === "string"
    && typeof event.data === "object"
    && event.data !== null
}
