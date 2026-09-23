import { useCallback, useEffect, useRef } from "react"
import { useQueryClient } from "@tanstack/react-query"
import { toast } from "sonner"
import { agentService } from "@/services/agent"
import { createMessageStreamRenderer } from "@/services/messageStreamRenderer"
import { streamAgentEvents } from "@/services/realtime"
import { queryKeys } from "@/hooks/useAgentQueries"
import { useAgentStore } from "@/stores/agentStore"
import type { AgentStep } from "@/types/agent"
import { isNodeTraceEvent, type AgentEvent, type AgentTerminalStatus } from "@/types/events"

interface SendOptions {
  message: string
  sessionId: string
}

const TERMINAL_STATUSES: ReadonlySet<string> = new Set(["completed", "failed", "stopped", "cancelled"])

function isTerminalStatus(status: unknown): status is AgentTerminalStatus {
  return typeof status === "string" && TERMINAL_STATUSES.has(status)
}

function stepFromLog(content: string, index: number): AgentStep {
  return {
    id: `log-${index}`,
    type: content.includes("检索") ? "retrieval" : "thinking",
    status: "success",
    title: content,
    duration: undefined,
  }
}

export function useAgentStream() {
  const queryClient = useQueryClient()
  const controllerRef = useRef<AbortController | null>(null)
  const activeSessionRef = useRef<string | null>(null)
  const cancelStreamRendererRef = useRef<() => void>(() => undefined)
  const seenLogsRef = useRef<Set<string>>(new Set())
  const {
    addMessage,
    updateMessage,
    setRunState,
    setSteps,
    upsertStep,
    applyNodeTraceEvent,
    resetNodeTraces,
  } = useAgentStore()

  const stop = useCallback(async () => {
    controllerRef.current?.abort()
    cancelStreamRendererRef.current()
    cancelStreamRendererRef.current = () => undefined
    const sessionId = activeSessionRef.current ?? useAgentStore.getState().currentSessionId
    const streamingMessage = [...useAgentStore.getState().messages]
      .reverse()
      .find((message) => message.status === "streaming")
    if (streamingMessage)
      updateMessage(streamingMessage.id, {
        content: streamingMessage.content || "执行已由用户停止。",
        status: "complete",
      })
    setRunState(false, "idle")
    try {
      await agentService.stop(sessionId)
      await queryClient.invalidateQueries({ queryKey: queryKeys.sessions })
      await queryClient.invalidateQueries({ queryKey: queryKeys.runs(sessionId) })
    } catch {
      toast.error("停止请求未被后端确认")
    }
    activeSessionRef.current = null
  }, [queryClient, setRunState, updateMessage])

  useEffect(
    () => () => {
      controllerRef.current?.abort()
      cancelStreamRendererRef.current()
    },
    [],
  )

  const handleEvent = useCallback(
    (event: AgentEvent, assistantId: string, logIndex: number) => {
      if (isNodeTraceEvent(event)) {
        applyNodeTraceEvent(event)
        return logIndex
      }
      if (event.type === "skill_event") {
        if (
          event.event === "skill_candidates" ||
          event.event === "skill_selected" ||
          event.event === "skill_llm_decision"
        ) {
          setRunState(true, "planning")
        } else if (event.event === "skill_tool_start") {
          setRunState(true, "running_tool")
        } else if (
          event.event === "skill_tool_end" ||
          event.event === "skill_completed" ||
          event.event === "skill_confirmation_required"
        ) {
          setRunState(true, "waiting_tool")
        }
        return logIndex
      }
      if (event.type === "meta") return logIndex
      if (event.type === "log" && event.content) {
        const content = event.content.trim()
        if (!content || seenLogsRef.current.has(content)) return logIndex
        seenLogsRef.current.add(content)
        upsertStep(event.step ?? stepFromLog(content, logIndex))
        return logIndex + 1
      }
      if (event.step) upsertStep(event.step)
      if (event.type === "planning") setRunState(true, "planning")
      if (event.type === "tool_call") setRunState(true, "running_tool")
      if (event.type === "tool_result") setRunState(true, "waiting_tool")
      if (event.type === "agent_complete") setRunState(false, "completed")
      if (event.type === "stopped") setRunState(false, "idle")
      if (event.type === "error" || event.type === "agent_error") {
        setRunState(false, "failed")
        updateMessage(assistantId, { content: event.content ?? "Agent 执行失败", status: "error" })
      }
      if (event.type === "terminal") {
        if (event.status === "failed") setRunState(false, "failed")
        else if (event.status === "stopped" || event.status === "cancelled") setRunState(false, "idle")
      }
      return logIndex
    },
    [applyNodeTraceEvent, setRunState, updateMessage, upsertStep],
  )

  const send = useCallback(
    async ({ message, sessionId }: SendOptions) => {
      controllerRef.current?.abort()
      cancelStreamRendererRef.current()
      cancelStreamRendererRef.current = () => undefined
      seenLogsRef.current.clear()
      setSteps([])
      resetNodeTraces()
      activeSessionRef.current = sessionId
      const userMessageId = crypto.randomUUID()
      const assistantId = crypto.randomUUID()
      addMessage({
        id: userMessageId,
        role: "user",
        content: message,
        createdAt: new Date().toISOString(),
        status: "complete",
      })
      addMessage({
        id: assistantId,
        role: "assistant",
        content: "",
        createdAt: new Date().toISOString(),
        status: "streaming",
      })
      setRunState(true, "thinking")
      const renderer = createMessageStreamRenderer((content, status) => {
        updateMessage(assistantId, { content, status })
      })
      cancelStreamRendererRef.current = renderer.cancel
      const controller = new AbortController()
      controllerRef.current = controller
      try {
        const userId = await agentService.getCurrentUserId()
        let logIndex = 0
        let terminalStatus: AgentTerminalStatus | null = null
        await streamAgentEvents(
          { message, user_id: userId, session_id: sessionId, request_id: crypto.randomUUID() },
          controller.signal,
          (event) => {
            if (event.type === "terminal" && isTerminalStatus(event.status)) terminalStatus = event.status
            if (event.type === "message_delta") {
              const hasDelta = typeof event.delta === "string"
              renderer.pushDelta(hasDelta ? (event.delta ?? "") : (event.content ?? ""), !hasDelta)
              return
            }
            if (event.type === "answer") {
              renderer.setFinalAnswer(event.content ?? event.delta ?? "")
              return
            }
            if (event.type === "error" || event.type === "agent_error") renderer.cancel()
            logIndex = handleEvent(event, assistantId, logIndex)
          },
          () => {
            renderer.close()
            void renderer.finished.then(() => {
              if (cancelStreamRendererRef.current !== renderer.cancel) return
              cancelStreamRendererRef.current = () => undefined
              activeSessionRef.current = null
              if (terminalStatus === "completed") {
                setRunState(false, "completed")
              } else if (terminalStatus === "failed") {
                setRunState(false, "failed")
              } else if (terminalStatus === "stopped" || terminalStatus === "cancelled") {
                setRunState(false, "idle")
              } else {
                setRunState(false, "failed")
                updateMessage(assistantId, {
                  content: "Agent 连接在返回终态前中断",
                  status: "error",
                  role: "error",
                })
              }
              void queryClient.invalidateQueries({ queryKey: queryKeys.sessions })
              void queryClient.invalidateQueries({ queryKey: queryKeys.history(sessionId) })
              void queryClient.invalidateQueries({ queryKey: queryKeys.runs(sessionId) })
            })
          },
        )
      } catch (error) {
        renderer.cancel()
        if (error instanceof DOMException && error.name === "AbortError") return
        activeSessionRef.current = null
        const content = error instanceof Error ? error.message : "Agent stream failed"
        updateMessage(assistantId, { content, status: "error", role: "error" })
        setRunState(false, "failed")
        toast.error(content)
      }
    },
    [addMessage, handleEvent, queryClient, resetNodeTraces, setRunState, setSteps, updateMessage],
  )

  return { send, stop }
}
