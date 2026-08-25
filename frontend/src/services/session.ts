import { apiClient } from "./api"
import type { Message, Session } from "@/types/chat"
import { agentService } from "./agent"

interface HistoryMessage { role: Message["role"]; content: string; created_at?: string }

export const sessionService = {
  async getSessions(): Promise<Session[]> {
    const { data } = await apiClient.get<{ sessions: Session[] }>("/sessions")
    return data.sessions
  },
  async getHistory(sessionId: string): Promise<Message[]> {
    const userId = await agentService.getCurrentUserId()
    const { data } = await apiClient.get<{ messages: HistoryMessage[] }>(`/history/${sessionId}`, { params: { user_id: userId } })
    return data.messages.map((message, index) => ({ id: `${sessionId}-${index}`, role: message.role, content: message.content, createdAt: message.created_at ?? new Date().toISOString(), status: "complete" }))
  },
  async deleteSession(sessionId: string): Promise<void> {
    await apiClient.delete(`/sessions/${sessionId}`)
  },
  async createSession(sessionId: string, agentId = "general"): Promise<Session> {
    const { data } = await apiClient.post<Session>("/sessions", { session_id: sessionId, agent_id: agentId })
    return data
  },
  async updateSession(sessionId: string, patch: { title?: string; favorite?: boolean }): Promise<Session> {
    const { data } = await apiClient.patch<Session>(`/sessions/${sessionId}`, patch)
    return data
  },
}
