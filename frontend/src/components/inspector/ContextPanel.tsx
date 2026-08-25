import * as Collapsible from "@radix-ui/react-collapsible"
import { ChevronDown, Database, Network } from "lucide-react"
import type { Message } from "@/types/chat"
import { useAgentStore } from "@/stores/agentStore"
import { NodeTraceCard } from "./NodeTraceCard"

function MessageCard({ message }: { message: Message }) {
  const label = message.role === "user" ? "用户消息" : message.role === "error" ? "Agent 错误" : "Agent 回复"
  return <Collapsible.Root>
    <Collapsible.Trigger className="group flex w-full items-center gap-2 rounded-md border bg-card p-2.5 text-left">
      <Database className="h-3.5 w-3.5 text-muted-foreground" />
      <span className="flex-1 text-xs font-medium">{label}</span>
      <span className="text-[9px] text-muted-foreground">约 {Math.max(1, Math.ceil(message.content.length / 2))} tok</span>
      <ChevronDown className="h-3 w-3 text-muted-foreground transition-transform group-data-[state=open]:rotate-180" />
    </Collapsible.Trigger>
    <Collapsible.Content>
      <pre className="mt-1 max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md border bg-muted/30 p-2.5 font-mono text-[10px] leading-4 text-muted-foreground">{message.content || "（等待响应）"}</pre>
    </Collapsible.Content>
  </Collapsible.Root>
}

export function ContextPanel() {
  const messages = useAgentStore((state) => state.messages)
  const nodeTraces = useAgentStore((state) => state.nodeTraces)
  if (messages.length === 0 && nodeTraces.length === 0) {
    return <p className="py-6 text-center text-xs text-muted-foreground">当前会话暂无上下文</p>
  }

  let latestUserIndex = -1
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "user") {
      latestUserIndex = index
      break
    }
  }
  const beforeTrace = latestUserIndex >= 0 ? messages.slice(0, latestUserIndex + 1) : []
  const afterTrace = latestUserIndex >= 0 ? messages.slice(latestUserIndex + 1) : messages

  return <div className="space-y-2">
    {beforeTrace.map((message) => <MessageCard key={message.id} message={message} />)}
    {nodeTraces.length > 0 && <section className="space-y-1.5">
      <div className="flex items-center gap-1.5 px-0.5 pt-1 text-[9px] font-semibold uppercase tracking-wider text-muted-foreground">
        <Network className="h-3 w-3" />实时 Node Trace · {nodeTraces.length}
      </div>
      {nodeTraces.map((trace) => <NodeTraceCard key={`${trace.turnId}:${trace.id}`} trace={trace} />)}
    </section>}
    {afterTrace.map((message) => <MessageCard key={message.id} message={message} />)}
  </div>
}
