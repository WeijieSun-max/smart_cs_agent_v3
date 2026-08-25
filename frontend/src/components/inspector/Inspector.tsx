import { Check, Circle, CircleDot, Clock3, Coins, ListChecks, Network, Wrench } from "lucide-react"
import { lazy, Suspense, useEffect, useMemo, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ErrorState } from "@/components/common/ErrorState"
import { FileTree } from "./FileTree"
import { ContextPanel } from "./ContextPanel"
import { useFiles, useRuns, useTools } from "@/hooks/useAgentQueries"
import { formatDuration } from "@/lib/utils"
import { useAgentStore } from "@/stores/agentStore"
import { useUiStore, type InspectorTab } from "@/stores/uiStore"

const WorkflowDiagram = lazy(() => import("@/components/agent/WorkflowDiagram"))

function ToolsPanel() {
  const { data, error, isLoading, refetch } = useTools()
  if (isLoading) return <div className="space-y-2">{Array.from({ length: 5 }, (_, i) => <Skeleton key={i} className="h-16" />)}</div>
  if (error) return <ErrorState message={error.message} onRetry={() => void refetch()} />
  return <div className="space-y-2">{data?.map((tool) => <div key={tool.name} className="rounded-md border bg-card p-2.5"><div className="flex items-center gap-2"><Wrench className="h-3.5 w-3.5 text-muted-foreground" /><code className="flex-1 font-mono text-[11px] font-medium">{tool.name}</code><span className="h-1.5 w-1.5 rounded-full bg-emerald-500" /></div><p className="mt-1.5 line-clamp-2 text-[10px] leading-4 text-muted-foreground">{tool.description}</p>{tool.category && <Badge variant="outline" className="mt-2">{tool.category}</Badge>}</div>)}</div>
}

function formatStepTime(value?: string) {
  if (!value) return "—"
  return new Date(value).toLocaleTimeString([], { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" })
}

function stepTokenLabel(step: { modelCalls?: number; tokenUsage?: { prompt: number; completion: number; total: number } }) {
  if (step.tokenUsage) return `${step.tokenUsage.total.toLocaleString()}（${step.tokenUsage.prompt.toLocaleString()} + ${step.tokenUsage.completion.toLocaleString()}）`
  return step.modelCalls ? "模型未返回 usage" : "—"
}

function TasksPanel() {
  const liveSteps = useAgentStore((state) => state.steps)
  const running = useAgentStore((state) => state.agentRunning)
  const currentSessionId = useAgentStore((state) => state.currentSessionId)
  const [selectedRunId, setSelectedRunId] = useState("")
  const { data: runs, isLoading, error, refetch } = useRuns(currentSessionId)
  useEffect(() => {
    setSelectedRunId("")
  }, [currentSessionId, running])
  const defaultRunId = running ? "live" : runs?.[0]?.turnId ?? "live"
  const activeRunId = selectedRunId || defaultRunId
  const selectedRun = runs?.find((run) => run.turnId === activeRunId)
  const showLive = activeRunId === "live" || !selectedRun
  const displayedSteps = showLive ? liveSteps : selectedRun.steps
  const displayedStatus = showLive ? "running" : selectedRun.status
  return <div>
    <div className="mb-3">
      <label className="mb-1.5 block text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">运行记录</label>
      <select value={activeRunId} onChange={(event) => setSelectedRunId(event.target.value === "live" ? "" : event.target.value)} disabled={isLoading} className="h-8 w-full rounded-md border bg-background px-2 text-xs outline-none focus:border-primary/50">
        {(running || !runs?.length) && <option value="live">{running ? "当前运行（实时）" : "暂无历史运行"}</option>}
        {runs?.map((run, index) => <option key={run.turnId} value={run.turnId}>{index === 0 ? "最近运行" : new Date(run.startedAt).toLocaleString()} · {run.status}</option>)}
      </select>
    </div>
    {error && <div className="mb-3 flex items-center justify-between gap-2 rounded-md border border-amber-500/20 bg-amber-500/5 px-2.5 py-2 text-[10px] text-muted-foreground"><span>历史运行暂时无法加载，实时流程不受影响</span><button className="shrink-0 text-primary hover:underline" onClick={() => void refetch()}>重试</button></div>}
    {displayedSteps.length === 0 ? <p className="py-4 text-center text-xs text-muted-foreground">发送消息后将实时显示执行步骤</p> : <div className="space-y-1.5">{displayedSteps.map((step) => <div key={step.id} className="rounded-md border bg-card px-2.5 py-2"><div className="flex items-start gap-2">{step.status === "success" ? <Check className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-500" /> : step.status === "running" ? <CircleDot className="mt-0.5 h-3.5 w-3.5 shrink-0 animate-pulse text-primary" /> : <Circle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />}<span className="min-w-0 flex-1 text-xs leading-4">{step.title}</span></div><div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1 pl-5 text-[9px] text-muted-foreground"><span title={`开始 ${step.startedAt ?? "—"}，结束 ${step.completedAt ?? "—"}`}><Clock3 className="mr-1 inline h-3 w-3" />{formatStepTime(step.startedAt)} → {formatStepTime(step.completedAt)} · {step.status === "running" ? "执行中" : formatDuration(step.duration)}</span><span><Coins className="mr-1 inline h-3 w-3" />Token {stepTokenLabel(step)}</span>{Boolean(step.modelCalls) && <span>模型调用 {step.modelCalls} 次</span>}</div></div>)}</div>}
    <div className="my-3 h-px bg-border" />
    <div className="mb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground"><Network className="h-3.5 w-3.5" />实际执行流程</div>
    <Suspense fallback={<Skeleton className="h-40" />}><WorkflowDiagram steps={displayedSteps} status={displayedStatus} /></Suspense>
    <p className="mt-2 text-[10px] text-muted-foreground">{showLive && running ? "正在实时记录" : "已持久化"} {displayedSteps.length} 个步骤</p>
  </div>
}

function DebugPanel() {
  const messages = useAgentStore((state) => state.messages)
  const currentSessionId = useAgentStore((state) => state.currentSessionId)
  const debug = useMemo(() => [...messages].reverse().find((message) => message.debug)?.debug, [messages])
  const usage = debug?.tokenUsage
  const rows = [["Request ID", debug?.requestId ?? "req_pending"], ["Session ID", currentSessionId], ["Trace ID", debug?.traceId ?? "—"], ["Model", debug?.model ?? "qwen3.7-flash"], ["Latency", formatDuration(debug?.latency)], ["Prompt tokens", usage?.prompt.toLocaleString() ?? "—"], ["Completion", usage?.completion.toLocaleString() ?? "—"], ["Tool calls", String(debug?.toolCalls ?? 0)]]
  const ratio = usage ? Math.min(100, (usage.total / usage.contextWindow) * 100) : 0
  return <div><div className="divide-y rounded-md border">{rows.map(([label, value]) => <div key={label} className="flex items-center justify-between gap-3 px-2.5 py-2"><span className="text-[10px] text-muted-foreground">{label}</span><code className="max-w-[155px] truncate font-mono text-[10px]">{value}</code></div>)}</div><div className="mt-3 rounded-md border p-2.5"><div className="flex items-center justify-between text-[10px]"><span className="text-muted-foreground">Context window</span><span>{usage?.total.toLocaleString() ?? 0} / {usage?.contextWindow.toLocaleString() ?? "32,768"}</span></div><div className="mt-2 h-1.5 overflow-hidden rounded-full bg-muted"><div className="h-full rounded-full bg-primary" style={{ width: `${ratio}%` }} /></div></div></div>
}

function FilesPanel() {
  const { data, error, isLoading, refetch } = useFiles()
  if (isLoading) return <Skeleton className="h-40" />
  if (error) return <ErrorState message={error.message} onRetry={() => void refetch()} />
  return <FileTree files={data ?? []} />
}

export function Inspector({ mobile = false }: { mobile?: boolean }) {
  const activeTab = useUiStore((state) => state.activeInspectorTab)
  const setActiveTab = useUiStore((state) => state.setActiveInspectorTab)
  return <aside className={`${mobile ? "w-full" : "w-[326px]"} flex h-full flex-col border-l bg-card`}><div className="flex h-14 shrink-0 items-center gap-2 border-b px-3"><ListChecks className="h-4 w-4" /><span className="text-sm font-semibold">Inspector</span><Badge variant="secondary" className="ml-auto">Live</Badge></div><Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as InspectorTab)} className="flex min-h-0 flex-1 flex-col"><div className="border-b px-2 py-2"><TabsList className="grid w-full grid-cols-5"><TabsTrigger value="context" className="px-1 text-[10px]">Context</TabsTrigger><TabsTrigger value="tools" className="px-1 text-[10px]">Tools</TabsTrigger><TabsTrigger value="tasks" className="px-1 text-[10px]">Tasks</TabsTrigger><TabsTrigger value="debug" className="px-1 text-[10px]">Debug</TabsTrigger><TabsTrigger value="files" className="px-1 text-[10px]">Files</TabsTrigger></TabsList></div><ScrollArea className="min-h-0 flex-1"><div className="p-3"><TabsContent value="context" className="m-0"><ContextPanel /></TabsContent><TabsContent value="tools" className="m-0"><ToolsPanel /></TabsContent><TabsContent value="tasks" className="m-0"><TasksPanel /></TabsContent><TabsContent value="debug" className="m-0"><DebugPanel /></TabsContent><TabsContent value="files" className="m-0"><FilesPanel /></TabsContent></div></ScrollArea></Tabs></aside>
}
