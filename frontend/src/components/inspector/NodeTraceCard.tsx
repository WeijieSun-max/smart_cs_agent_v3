import * as Collapsible from "@radix-ui/react-collapsible"
import { AlertCircle, CheckCircle2, ChevronDown, CircleDot, Clock3, Cpu } from "lucide-react"
import { useEffect, useState } from "react"
import { cn, formatDuration } from "@/lib/utils"
import type { NodeTrace } from "@/types/nodeTrace"
import { NodeTraceValue } from "./NodeTraceValue"

function duration(trace: NodeTrace) {
  if (trace.status === "running") return "运行中"
  if (!trace.startedAt || !trace.completedAt) return "—"
  return formatDuration(Math.max(0, new Date(trace.completedAt).getTime() - new Date(trace.startedAt).getTime()))
}

type TraceValueKind = "data" | "prompt" | "response"

function TraceSection({
  title,
  value,
  kind = "data",
  defaultOpen = false,
}: {
  title: string
  value: unknown
  kind?: TraceValueKind
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return <Collapsible.Root open={open} onOpenChange={setOpen}>
    <Collapsible.Trigger className="group flex w-full items-center gap-2 py-1 text-left">
      <ChevronDown className="h-3 w-3 shrink-0 text-muted-foreground transition-transform group-data-[state=open]:rotate-180" />
      <span className="text-[10px] font-semibold text-muted-foreground">{title}</span>
    </Collapsible.Trigger>
    <Collapsible.Content className="pb-1"><NodeTraceValue value={value} kind={kind} /></Collapsible.Content>
  </Collapsible.Root>
}

export function NodeTraceCard({ trace }: { trace: NodeTrace }) {
  const [open, setOpen] = useState(trace.status === "running")
  useEffect(() => {
    if (trace.status === "running") setOpen(true)
  }, [trace.status])

  const StatusIcon = trace.status === "success" ? CheckCircle2 : trace.status === "error" ? AlertCircle : CircleDot
  return <Collapsible.Root open={open} onOpenChange={setOpen} className={cn(
    "overflow-hidden rounded-md border bg-card",
    trace.status === "running" && "border-primary/40",
    trace.status === "error" && "border-red-500/40",
  )}>
    <Collapsible.Trigger className="group flex w-full items-start gap-2 p-2.5 text-left">
      <StatusIcon className={cn(
        "mt-0.5 h-3.5 w-3.5 shrink-0",
        trace.status === "running" && "animate-pulse text-primary",
        trace.status === "success" && "text-emerald-500",
        trace.status === "error" && "text-red-500",
      )} />
      <span className="min-w-0 flex-1">
        <span className="block truncate font-mono text-[11px] font-semibold" title={trace.nodeName}>{trace.nodeName}</span>
        <span className="mt-1 flex flex-wrap gap-x-2 text-[9px] text-muted-foreground">
          <span><Clock3 className="mr-1 inline h-2.5 w-2.5" />{duration(trace)}</span>
          <span><Cpu className="mr-1 inline h-2.5 w-2.5" />LLM {trace.modelCalls.length}</span>
        </span>
      </span>
      <ChevronDown className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform group-data-[state=open]:rotate-180" />
    </Collapsible.Trigger>
    <Collapsible.Content>
      <div className="space-y-1 border-t px-2.5 py-2">
        <TraceSection title="Node Input" value={trace.input} />
        {trace.modelCalls.map((call, index) => <div key={call.id} className="rounded border bg-background/50 px-2 py-1">
          <div className="mb-0.5 flex items-center justify-between text-[9px] font-semibold text-muted-foreground">
            <span>LLM Call #{index + 1}</span>
            <span className={cn(call.status === "error" && "text-red-500", call.status === "success" && "text-emerald-500")}>{call.status}</span>
          </div>
          <TraceSection title="LLM Prompt" value={call.prompt} kind="prompt" />
          <TraceSection title="LLM Response" value={call.response} kind="response" />
        </div>)}
        <TraceSection title="Node Output" value={trace.output} />
        {trace.error !== undefined && <TraceSection title="Node Error" value={trace.error} defaultOpen />}
      </div>
    </Collapsible.Content>
  </Collapsible.Root>
}
