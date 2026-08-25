import { Bot, Sparkles } from "lucide-react"
import { useEffect, useRef } from "react"
import { EmptyState } from "@/components/common/EmptyState"
import { MessageItem } from "./MessageItem"
import { ChatComposer } from "./ChatComposer"
import { ChatHeader } from "./ChatHeader"
import { useAgentStream } from "@/hooks/useAgentStream"
import { useAgentStore } from "@/stores/agentStore"

export function ChatPanel() {
  const messages = useAgentStore((state) => state.messages)
  const sessionId = useAgentStore((state) => state.currentSessionId)
  const running = useAgentStore((state) => state.agentRunning)
  const status = useAgentStore((state) => state.agentStatus)
  const scrollRef = useRef<HTMLDivElement>(null)
  const { send, stop } = useAgentStream()
  useEffect(() => { scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" }) }, [messages, running])
  const statusText = status === "planning" ? "Planning execution..." : status === "running_tool" ? "Running tool..." : status === "waiting_tool" ? "Waiting for result..." : "Thinking..."
  return <section className="flex h-full min-w-0 flex-col"><ChatHeader /><div ref={scrollRef} className="surface-grid min-h-0 flex-1 overflow-y-auto">
    {messages.length === 0 ? <EmptyState title="开始一项 Agent 任务" description="询问问题、分析项目或执行多步骤任务。运行过程会以时间线实时展示。" icon={Bot} /> : messages.map((message) => <MessageItem key={message.id} message={message} />)}
    {running && <div className="mx-auto flex w-full max-w-3xl items-center gap-2 px-8 pb-4 text-xs text-primary"><Sparkles className="h-3.5 w-3.5 animate-pulse" /><span>{statusText}</span><span className="flex gap-1"><i className="h-1 w-1 animate-pulse-dot rounded-full bg-primary" /><i className="h-1 w-1 animate-pulse-dot rounded-full bg-primary [animation-delay:150ms]" /><i className="h-1 w-1 animate-pulse-dot rounded-full bg-primary [animation-delay:300ms]" /></span></div>}
  </div><ChatComposer onSend={(message) => void send({ message, sessionId })} onStop={() => void stop()} /></section>
}
