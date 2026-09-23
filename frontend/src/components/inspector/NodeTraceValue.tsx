import { Check, Copy } from "lucide-react"
import { useMemo, useState } from "react"
import { toast } from "sonner"
import { cn } from "@/lib/utils"

type TraceValueKind = "data" | "prompt" | "response"

interface TraceMessage {
  type: string
  content: unknown
}

const MESSAGE_TYPES = new Set([
  "ai",
  "assistant",
  "chat",
  "function",
  "human",
  "system",
  "tool",
  "user",
])

const MESSAGE_LABELS: Record<string, string> = {
  ai: "ASSISTANT",
  assistant: "ASSISTANT",
  human: "USER",
  user: "USER",
  system: "SYSTEM",
  tool: "TOOL",
  function: "FUNCTION",
  chat: "CHAT",
}

const MESSAGE_STYLES: Record<string, string> = {
  ai: "border-emerald-500/30 bg-emerald-500/5 text-emerald-700 dark:text-emerald-300",
  assistant: "border-emerald-500/30 bg-emerald-500/5 text-emerald-700 dark:text-emerald-300",
  human: "border-blue-500/30 bg-blue-500/5 text-blue-700 dark:text-blue-300",
  user: "border-blue-500/30 bg-blue-500/5 text-blue-700 dark:text-blue-300",
  system: "border-amber-500/30 bg-amber-500/5 text-amber-700 dark:text-amber-300",
  tool: "border-violet-500/30 bg-violet-500/5 text-violet-700 dark:text-violet-300",
  function: "border-violet-500/30 bg-violet-500/5 text-violet-700 dark:text-violet-300",
}

function formatValue(value: unknown) {
  if (value === undefined) return "（尚未返回）"
  if (typeof value === "string") return value
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function findMessages(value: unknown): TraceMessage[] {
  const messages: TraceMessage[] = []
  const visited = new Set<object>()

  const visit = (candidate: unknown, depth: number) => {
    if (depth > 12 || messages.length >= 200 || candidate === null || typeof candidate !== "object") return
    if (visited.has(candidate)) return
    visited.add(candidate)

    if (isRecord(candidate)) {
      const type = typeof candidate.type === "string" ? candidate.type.toLowerCase() : ""
      if (MESSAGE_TYPES.has(type) && Object.hasOwn(candidate, "content")) {
        messages.push({ type, content: candidate.content })
        return
      }
      Object.values(candidate).forEach((item) => visit(item, depth + 1))
      return
    }
    if (Array.isArray(candidate)) {
      candidate.forEach((item) => visit(item, depth + 1))
    }
  }

  visit(value, 0)
  return messages
}

function byteSizeLabel(text: string) {
  const bytes = new TextEncoder().encode(text).length
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function MessageList({ messages }: { messages: TraceMessage[] }) {
  return <div className="max-h-96 space-y-1.5 overflow-auto p-1.5">
    {messages.map((message, index) => <div key={`${message.type}-${index}`} className="rounded border bg-background/70">
      <div className="border-b px-2 py-1">
        <span className={cn(
          "inline-flex rounded border px-1.5 py-0.5 font-mono text-[8px] font-semibold tracking-wide",
          MESSAGE_STYLES[message.type] ?? "border-border bg-muted text-muted-foreground",
        )}>{MESSAGE_LABELS[message.type] ?? message.type.toUpperCase()}</span>
      </div>
      <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words px-2 py-1.5 font-mono text-[10px] leading-4 text-foreground/80">{formatValue(message.content)}</pre>
    </div>)}
  </div>
}

export function NodeTraceValue({ value, kind = "data" }: { value: unknown; kind?: TraceValueKind }) {
  const [copied, setCopied] = useState(false)
  const [rawView, setRawView] = useState(false)
  const text = useMemo(() => formatValue(value), [value])
  const messages = useMemo(() => kind === "data" ? [] : findMessages(value), [kind, value])
  const showMessages = messages.length > 0 && !rawView
  const redacted = text.includes("[REDACTED_")
  const format = value === undefined ? "PENDING" : typeof value === "string" ? "TEXT" : "JSON"

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1400)
    } catch {
      toast.error("复制失败，请手动选择内容")
    }
  }

  return <div>
    <div className="overflow-hidden rounded-md border bg-muted/20">
      <div className="flex min-h-7 items-center gap-1 border-b bg-background/60 px-1.5">
        {messages.length > 0 && <div className="flex rounded bg-muted/60 p-0.5">
          <button
            type="button"
            className={cn("rounded px-1.5 py-0.5 text-[8px]", !rawView ? "bg-background font-semibold text-foreground shadow-sm" : "text-muted-foreground")}
            onClick={() => setRawView(false)}
            aria-pressed={!rawView}
          >消息 {messages.length}</button>
          <button
            type="button"
            className={cn("rounded px-1.5 py-0.5 text-[8px]", rawView ? "bg-background font-semibold text-foreground shadow-sm" : "text-muted-foreground")}
            onClick={() => setRawView(true)}
            aria-pressed={rawView}
          >原始 JSON</button>
        </div>}
        <span className="ml-auto font-mono text-[8px] text-muted-foreground">{format} · {byteSizeLabel(text)}</span>
        <button
          type="button"
          className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
          onClick={() => void copy()}
          aria-label="复制原始数据"
          title="复制原始数据"
        >
          {copied ? <Check className="h-3 w-3 text-emerald-500" /> : <Copy className="h-3 w-3" />}
        </button>
      </div>
      {showMessages
        ? <MessageList messages={messages} />
        : <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words p-2 font-mono text-[10px] leading-4 text-foreground/80">{text}</pre>}
    </div>
    {redacted && <span className="mt-1 block text-[9px] text-amber-600 dark:text-amber-400">凭据或受保护的记忆内容已脱敏，其余为原始追踪数据</span>}
  </div>
}
