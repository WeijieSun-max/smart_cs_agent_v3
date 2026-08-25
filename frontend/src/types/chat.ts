import type { AgentStep, DebugInfo } from "./agent"

export type MessageRole = "user" | "assistant" | "tool" | "error"
export type MessageStatus = "sending" | "streaming" | "complete" | "error"

export interface Message {
  id: string
  role: MessageRole
  content: string
  createdAt: string
  status?: MessageStatus
  steps?: AgentStep[]
  debug?: DebugInfo
}

export interface Session {
  id: string
  title: string
  agentId: string
  createdAt: string
  updatedAt: string
  favorite: boolean
  messageCount: number
}

export interface ChatRunRequest {
  message: string
  user_id?: string
  session_id?: string
  request_id?: string
}

export interface ChatResponse {
  response: string
  session_id: string
  turn_id: string
  trace_id: string
  intent: string
  compliance_passed: boolean
}
