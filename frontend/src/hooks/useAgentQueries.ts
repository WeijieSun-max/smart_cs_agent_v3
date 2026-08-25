import { useQuery } from "@tanstack/react-query"
import { agentService } from "@/services/agent"
import { sessionService } from "@/services/session"

export const queryKeys = {
  agents: ["agents"] as const,
  tools: ["tools"] as const,
  files: ["files"] as const,
  sessions: ["sessions"] as const,
  history: (id: string) => ["sessions", id, "history"] as const,
  runs: (id: string) => ["sessions", id, "runs"] as const,
}

export function useAgents() { return useQuery({ queryKey: queryKeys.agents, queryFn: agentService.getAgents, staleTime: 60_000 }) }
export function useTools() { return useQuery({ queryKey: queryKeys.tools, queryFn: agentService.getTools, staleTime: 60_000 }) }
export function useFiles() { return useQuery({ queryKey: queryKeys.files, queryFn: agentService.getFiles, staleTime: 30_000 }) }
export function useSessions() { return useQuery({ queryKey: queryKeys.sessions, queryFn: sessionService.getSessions, staleTime: 15_000 }) }
export function useHistory(sessionId: string) { return useQuery({ queryKey: queryKeys.history(sessionId), queryFn: () => sessionService.getHistory(sessionId), enabled: Boolean(sessionId) }) }
export function useRuns(sessionId: string) { return useQuery({ queryKey: queryKeys.runs(sessionId), queryFn: () => agentService.getRuns(sessionId), enabled: Boolean(sessionId), staleTime: 5_000 }) }
