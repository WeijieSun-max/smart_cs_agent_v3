import type { ChatRunRequest } from "@/types/chat"
import type { AgentEvent } from "@/types/events"

const baseUrl = import.meta.env.VITE_API_BASE_URL ?? "/api"

export async function streamAgentEvents(
  request: ChatRunRequest,
  signal: AbortSignal,
  onEvent: (event: AgentEvent) => void,
  onDone: (doneMarkerReceived: boolean) => void,
) {
  const response = await fetch(`${baseUrl}/chat_stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(request),
    signal,
  })
  if (!response.ok || !response.body) throw new Error(`Streaming request failed (${response.status})`)
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let doneMarkerReceived = false
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const frames = buffer.split("\n\n")
    buffer = frames.pop() ?? ""
    for (const frame of frames) {
      const payload = frame.split("\n").filter((line) => line.startsWith("data:")).map((line) => line.slice(5)).join("\n")
      if (!payload) continue
      if (payload === "[DONE]") { doneMarkerReceived = true; continue }
      try { onEvent(JSON.parse(payload) as AgentEvent) } catch { /* Ignore heartbeat or malformed frames. */ }
    }
  }
  onDone(doneMarkerReceived)
}

export class AgentWebSocketClient {
  private socket: WebSocket | null = null

  connect(sessionId: string, onEvent: (event: AgentEvent) => void) {
    const origin = window.location.origin.replace(/^http/, "ws")
    this.socket = new WebSocket(`${origin}${baseUrl}/agent/ws/${sessionId}`)
    this.socket.addEventListener("message", (event: MessageEvent<string>) => {
      try { onEvent(JSON.parse(event.data) as AgentEvent) } catch { /* Ignore invalid server frames. */ }
    })
  }

  disconnect() { this.socket?.close(); this.socket = null }
  send(payload: Record<string, unknown>) { this.socket?.send(JSON.stringify(payload)) }
}
