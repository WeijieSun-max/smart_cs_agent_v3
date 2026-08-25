import { create } from "zustand"
import type { AgentStatus, AgentStep } from "@/types/agent"
import type { Message } from "@/types/chat"
import type { NodeTrace, NodeTraceEvent } from "@/types/nodeTrace"
import { reduceNodeTraceEvent } from "@/stores/nodeTraceReducer"

interface AgentState {
  currentAgentId: string
  currentSessionId: string
  agentStatus: AgentStatus
  agentRunning: boolean
  messages: Message[]
  steps: AgentStep[]
  nodeTraces: NodeTrace[]
  seenNodeTraceEvents: Record<string, true>
  setCurrentAgent: (id: string) => void
  setCurrentSession: (id: string) => void
  setRunState: (running: boolean, status?: AgentStatus) => void
  setMessages: (messages: Message[]) => void
  addMessage: (message: Message) => void
  updateMessage: (id: string, patch: Partial<Message>) => void
  setSteps: (steps: AgentStep[]) => void
  upsertStep: (step: AgentStep) => void
  applyNodeTraceEvent: (event: NodeTraceEvent) => void
  resetNodeTraces: () => void
  clearContext: () => void
}

export const useAgentStore = create<AgentState>((set) => ({
  currentAgentId: "general",
  currentSessionId: "session_demo",
  agentStatus: "completed",
  agentRunning: false,
  messages: [],
  steps: [],
  nodeTraces: [],
  seenNodeTraceEvents: {},
  setCurrentAgent: (currentAgentId) => set({ currentAgentId }),
  setCurrentSession: (currentSessionId) => set((state) => state.currentSessionId === currentSessionId
    ? { currentSessionId }
    : { currentSessionId, nodeTraces: [], seenNodeTraceEvents: {} }),
  setRunState: (agentRunning, agentStatus = agentRunning ? "thinking" : "idle") => set({ agentRunning, agentStatus }),
  setMessages: (messages) => set({ messages }),
  addMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
  updateMessage: (id, patch) => set((state) => ({ messages: state.messages.map((message) => message.id === id ? { ...message, ...patch } : message) })),
  setSteps: (steps) => set({ steps }),
  upsertStep: (step) => set((state) => {
    const exists = state.steps.some((item) => item.id === step.id)
    return { steps: exists ? state.steps.map((item) => item.id === step.id ? { ...item, ...step } : item) : [...state.steps, step] }
  }),
  applyNodeTraceEvent: (event) => set((state) => reduceNodeTraceEvent(state, event)),
  resetNodeTraces: () => set({ nodeTraces: [], seenNodeTraceEvents: {} }),
  clearContext: () => set({ messages: [], steps: [], nodeTraces: [], seenNodeTraceEvents: {}, agentStatus: "idle", agentRunning: false }),
}))
