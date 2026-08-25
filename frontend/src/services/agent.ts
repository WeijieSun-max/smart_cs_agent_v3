import { apiClient } from "./api"
import type { Agent, AgentRunRecord, AgentStep, Tool } from "@/types/agent"
import type { ChatRunRequest, ChatResponse } from "@/types/chat"
import type { WorkspaceFile } from "@/types/workspace"

interface BackendTool {
  name: string
  description: string
  category?: string
  inputSchema?: Record<string, unknown>
  enabled?: boolean
  status?: Tool["status"]
}
interface BackendRun {
  turn_id: string
  session_id: string
  status: AgentRunRecord["status"]
  stop_requested: boolean
  started_at: string
  completed_at?: string
  steps: AgentStep[]
}

export const agentService = {
  async getCurrentUserId(): Promise<string> {
    const { data } = await apiClient.get<{ user_id: string }>("/settings/current-user")
    return data.user_id
  },
  async updateCurrentUserId(userId: string): Promise<string> {
    const { data } = await apiClient.put<{ user_id: string }>("/settings/current-user", { user_id: userId })
    return data.user_id
  },
  async getAgents(): Promise<Agent[]> {
    const { data } = await apiClient.get<{ agents: Agent[] }>("/agents")
    return data.agents
  },
  async getTools(): Promise<Tool[]> {
    const { data } = await apiClient.get<{ tools: BackendTool[] }>("/tools")
    return data.tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      category: tool.category,
      inputSchema: tool.inputSchema,
      enabled: tool.enabled ?? true,
      status: tool.status ?? "available",
    }))
  },
  async getFiles(): Promise<WorkspaceFile[]> {
    const { data } = await apiClient.get<{ files: WorkspaceFile[] }>("/files")
    return data.files
  },
  async run(request: ChatRunRequest): Promise<ChatResponse> {
    const { data } = await apiClient.post<ChatResponse>("/chat", request)
    return data
  },
  async stop(sessionId: string): Promise<void> {
    await apiClient.post("/agent/stop", { session_id: sessionId })
  },
  async getRuns(sessionId: string): Promise<AgentRunRecord[]> {
    const { data } = await apiClient.get<{ runs: BackendRun[] }>("/runs", {
      params: { session_id: sessionId },
    })
    return data.runs.map((run) => ({
      turnId: run.turn_id,
      sessionId: run.session_id,
      status: run.status,
      stopRequested: run.stop_requested,
      startedAt: run.started_at,
      completedAt: run.completed_at,
      steps: run.steps,
    }))
  },
}
