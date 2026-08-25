import * as Collapsible from "@radix-ui/react-collapsible"
import { Check, ChevronDown, Circle, Clock3, Search, Terminal, Wrench, X } from "lucide-react"
import { lazy, memo, Suspense, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { cn, formatDuration, safeJson } from "@/lib/utils"
import type { AgentStep, StepStatus } from "@/types/agent"

const CodeBlock = lazy(() => import("@/components/chat/CodeBlock").then((module) => ({ default: module.CodeBlock })))

const statusVariant: Record<StepStatus, "secondary" | "default" | "success" | "destructive"> = { pending: "secondary", running: "default", success: "success", error: "destructive" }
const icons = { query: Circle, plan: Circle, thinking: Circle, tool_call: Wrench, tool_result: Search, retrieval: Search, sandbox: Terminal, final: Check, error: X }

const TimelineStep = memo(function TimelineStep({ step, last }: { step: AgentStep; last: boolean }) {
  const [open, setOpen] = useState(step.type === "tool_call" || step.type === "sandbox")
  const Icon = icons[step.type]
  const expandable = step.input !== undefined || step.output !== undefined || Boolean(step.description)
  return <div className="relative flex gap-3 pb-3">
    {!last && <div className="absolute left-[13px] top-7 h-[calc(100%-16px)] w-px bg-border" />}
    <div className={cn("z-10 mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border bg-background", step.status === "running" && "border-primary text-primary", step.status === "success" && "text-emerald-500", step.status === "error" && "text-red-500")}><Icon className={cn("h-3.5 w-3.5", step.status === "running" && "animate-pulse")} /></div>
    <Collapsible.Root open={open} onOpenChange={setOpen} className="min-w-0 flex-1">
      <Collapsible.Trigger disabled={!expandable} className="flex w-full items-center gap-2 py-1 text-left"><span className="min-w-0 flex-1 truncate text-xs font-medium">{step.title}</span>{step.toolName && <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-[9px] text-muted-foreground">{step.toolName}</code>}<Badge variant={statusVariant[step.status]}>{step.status}</Badge>{step.duration !== undefined && <span className="flex items-center gap-1 text-[10px] text-muted-foreground"><Clock3 className="h-3 w-3" />{formatDuration(step.duration)}</span>}{expandable && <ChevronDown className={cn("h-3.5 w-3.5 text-muted-foreground transition-transform", open && "rotate-180")} />}</Collapsible.Trigger>
      <Collapsible.Content className="pt-1"><div className="rounded-md border bg-muted/20 p-2.5">{step.description && <p className="text-xs leading-5 text-muted-foreground">{step.description}</p>}{step.input !== undefined && <div><p className="mt-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Input</p><Suspense fallback={<pre className="mt-2 overflow-auto font-mono text-[10px]">{safeJson(step.input)}</pre>}><CodeBlock language="json" code={safeJson(step.input)} /></Suspense></div>}{step.output !== undefined && <div><p className="mt-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Output</p><Suspense fallback={<pre className="mt-2 overflow-auto font-mono text-[10px]">{safeJson(step.output)}</pre>}><CodeBlock language="json" code={safeJson(step.output)} /></Suspense></div>}</div></Collapsible.Content>
    </Collapsible.Root>
  </div>
})

export function ExecutionTimeline({ steps }: { steps: AgentStep[] }) {
  const [open, setOpen] = useState(true)
  if (steps.length === 0) return null
  return <Collapsible.Root open={open} onOpenChange={setOpen} className="mt-4 border-t pt-3">
    <Collapsible.Trigger className="group mb-3 flex w-full items-center gap-2 rounded-sm text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40">
      <span className="flex-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Execution trace · {steps.length} steps</span>
      <span className="text-[10px] text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100">{open ? "收起" : "展开"}</span>
      <ChevronDown className={cn("h-3.5 w-3.5 text-muted-foreground transition-transform", open && "rotate-180")} />
    </Collapsible.Trigger>
    <Collapsible.Content>{steps.map((step, index) => <TimelineStep key={step.id} step={step} last={index === steps.length - 1} />)}</Collapsible.Content>
  </Collapsible.Root>
}
