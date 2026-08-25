export type AgentStatus = "idle" | "thinking" | "planning" | "running_tool" | "waiting_tool" | "completed" | "failed"
export type StepStatus = "pending" | "running" | "success" | "error"
export type StepType = "query" | "plan" | "thinking" | "tool_call" | "tool_result" | "retrieval" | "sandbox" | "final" | "error"

export interface Tool {
  name: string
  description: string
  enabled: boolean
  category?: string
  status?: "available" | "busy" | "offline"
  inputSchema?: Record<string, unknown>
}

export interface Agent {
  id: string
  name: string
  description: string
  status: AgentStatus
  tools: Tool[]
  model: string
  icon: "sparkles" | "code" | "search" | "chart"
}

export interface AgentStep {
  id: string
  type: StepType
  status: StepStatus
  title: string
  description?: string
  toolName?: string
  input?: unknown
  output?: unknown
  duration?: number
  startedAt?: string
  completedAt?: string
  modelCalls?: number
  tokenUsage?: {
    prompt: number
    completion: number
    total: number
  }
}

export interface AgentRunRecord {
  turnId: string
  sessionId: string
  status: "running" | "completed" | "failed" | "stopped" | "cancelled"
  stopRequested: boolean
  startedAt: string
  completedAt?: string
  steps: AgentStep[]
}

export interface TokenUsage {
  prompt: number
  completion: number
  total: number
  contextWindow: number
}

export interface DebugInfo {
  requestId: string
  sessionId: string
  traceId?: string
  model: string
  latency: number
  tokenUsage: TokenUsage
  toolCalls: number
}
