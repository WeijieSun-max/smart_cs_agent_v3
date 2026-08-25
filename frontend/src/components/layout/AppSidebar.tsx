import * as ContextMenu from "@radix-ui/react-context-menu"
import * as DropdownMenu from "@radix-ui/react-dropdown-menu"
import { BarChart3, Bot, ChevronsRight, Code2, MoreHorizontal, PanelLeftClose, Plus, Search, Settings, Sparkles, Star, Trash2 } from "lucide-react"
import { useMemo, useState } from "react"
import { useQueryClient } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { LoadingState } from "@/components/common/LoadingState"
import { ErrorState } from "@/components/common/ErrorState"
import { queryKeys, useAgents, useSessions } from "@/hooks/useAgentQueries"
import { cn, formatRelativeTime } from "@/lib/utils"
import { sessionService } from "@/services/session"
import { useAgentStore } from "@/stores/agentStore"
import { useUiStore } from "@/stores/uiStore"

const agentIcons = { sparkles: Sparkles, code: Code2, search: Search, chart: BarChart3 }

export function AppSidebar({ mobile = false }: { mobile?: boolean }) {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const collapsed = useUiStore((state) => state.sidebarCollapsed) && !mobile
  const toggleSidebar = useUiStore((state) => state.toggleSidebar)
  const setMobileOpen = useUiStore((state) => state.setSidebarMobileOpen)
  const currentAgentId = useAgentStore((state) => state.currentAgentId)
  const currentSessionId = useAgentStore((state) => state.currentSessionId)
  const setCurrentAgent = useAgentStore((state) => state.setCurrentAgent)
  const setCurrentSession = useAgentStore((state) => state.setCurrentSession)
  const clearContext = useAgentStore((state) => state.clearContext)
  const agentRunning = useAgentStore((state) => state.agentRunning)
  const [search, setSearch] = useState("")
  const [creatingSession, setCreatingSession] = useState(false)
  const { data: agents, isLoading: agentsLoading, error: agentsError, refetch: refetchAgents } = useAgents()
  const { data: sessions, isLoading: sessionsLoading, error: sessionsError, refetch: refetchSessions } = useSessions()
  const visibleSessions = useMemo(() => {
    const preferredEmptyByAgent = new Map<string, string>()
    for (const session of sessions ?? []) {
      if (session.messageCount !== 0) continue
      if (session.id === currentSessionId || !preferredEmptyByAgent.has(session.agentId)) {
        preferredEmptyByAgent.set(session.agentId, session.id)
      }
    }
    return (sessions ?? []).filter((session) => session.messageCount !== 0 || preferredEmptyByAgent.get(session.agentId) === session.id)
  }, [currentSessionId, sessions])
  const filteredSessions = useMemo(() => visibleSessions.filter((session) => session.title.toLowerCase().includes(search.toLowerCase())), [search, visibleSessions])

  const selectSession = (id: string) => { setCurrentSession(id); navigate(`/chat/${id}`); setMobileOpen(false) }
  const newChat = async () => {
    if (creatingSession || agentRunning) return
    const reusableSession = sessions?.find((session) => (
      session.id === currentSessionId
      && session.agentId === currentAgentId
      && session.messageCount === 0
    )) ?? sessions?.find((session) => session.agentId === currentAgentId && session.messageCount === 0)
    if (reusableSession) {
      setCurrentSession(reusableSession.id)
      clearContext()
      navigate(`/chat/${reusableSession.id}`)
      setMobileOpen(false)
      return
    }
    setCreatingSession(true)
    const id = crypto.randomUUID()
    try {
      const createdSession = await sessionService.createSession(id, currentAgentId)
      await queryClient.invalidateQueries({ queryKey: queryKeys.sessions })
      setCurrentSession(createdSession.id)
      clearContext()
      navigate(`/chat/${createdSession.id}`)
      setMobileOpen(false)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "创建会话失败")
    } finally {
      setCreatingSession(false)
    }
  }
  const deleteSession = async (id: string) => {
    try {
      await sessionService.deleteSession(id)
      await queryClient.invalidateQueries({ queryKey: queryKeys.sessions })
      if (id === currentSessionId) await newChat()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "删除会话失败")
    }
  }
  const toggleFavorite = async (id: string, favorite: boolean) => {
    try {
      await sessionService.updateSession(id, { favorite: !favorite })
      await queryClient.invalidateQueries({ queryKey: queryKeys.sessions })
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "更新会话失败")
    }
  }

  return <aside className={cn("flex h-full shrink-0 flex-col border-r bg-card transition-[width] duration-200", collapsed ? "w-[62px]" : "w-[260px]", mobile && "w-full border-r-0")}>
    <div className="flex h-14 items-center gap-2 border-b px-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-foreground text-background"><Bot className="h-4 w-4" /></div>
      {!collapsed && <div className="min-w-0 flex-1"><p className="truncate text-sm font-semibold">Agent Workbench</p><p className="text-[10px] text-muted-foreground">Developer Console</p></div>}
      {!mobile && <Button variant="ghost" size="icon-sm" onClick={toggleSidebar} aria-label="折叠侧栏">{collapsed ? <ChevronsRight className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}</Button>}
    </div>
    <div className="p-2"><Tooltip><TooltipTrigger asChild><Button className={cn("w-full", collapsed ? "px-0" : "justify-start")} onClick={() => void newChat()} disabled={creatingSession || agentRunning}><Plus className="h-4 w-4" />{!collapsed && (creatingSession ? "创建中..." : "新建会话")}</Button></TooltipTrigger>{collapsed && <TooltipContent side="right">{agentRunning ? "请先停止当前 Agent" : "新建会话"}</TooltipContent>}</Tooltip></div>
    <ScrollArea className="min-h-0 flex-1">
      {!collapsed && <div className="px-3 pb-1 pt-3 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Agents</div>}
      {agentsLoading ? <LoadingState rows={3} /> : agentsError ? <ErrorState message={agentsError.message} onRetry={() => void refetchAgents()} /> : <div className="space-y-0.5 px-2">
        {agents?.map((agent) => { const Icon = agentIcons[agent.icon]; const selected = agent.id === currentAgentId; return <Tooltip key={agent.id}><TooltipTrigger asChild><button className={cn("flex w-full items-center gap-2.5 rounded-md px-2 py-2 text-left text-xs transition-colors hover:bg-muted", selected && "bg-accent text-accent-foreground", collapsed && "justify-center px-0")} onClick={() => setCurrentAgent(agent.id)}><Icon className="h-4 w-4 shrink-0" />{!collapsed && <><span className="min-w-0 flex-1 truncate">{agent.name}</span><span className={cn("h-1.5 w-1.5 rounded-full", selected ? "bg-primary" : "bg-foreground/20")} /></>}</button></TooltipTrigger>{collapsed && <TooltipContent side="right">{agent.name}</TooltipContent>}</Tooltip> })}
      </div>}
      {!collapsed && <>
        <div className="mt-4 flex items-center justify-between px-3 pb-2"><span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Sessions</span><Star className="h-3 w-3 text-muted-foreground" /></div>
        <div className="px-2 pb-2"><div className="flex items-center gap-2 rounded-md border bg-background px-2"><Search className="h-3.5 w-3.5 text-muted-foreground" /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索会话..." className="h-8 min-w-0 flex-1 bg-transparent text-xs outline-none placeholder:text-muted-foreground" /></div></div>
        {sessionsLoading ? <LoadingState rows={4} /> : sessionsError ? <ErrorState message={sessionsError.message} onRetry={() => void refetchSessions()} /> : filteredSessions.length === 0 ? <p className="px-3 py-6 text-center text-xs text-muted-foreground">暂无会话，点击上方按钮创建</p> : <div className="space-y-0.5 px-2 pb-4">{filteredSessions.map((session) => <ContextMenu.Root key={session.id}><ContextMenu.Trigger asChild><button onClick={() => selectSession(session.id)} className={cn("group flex w-full items-start gap-2 rounded-md px-2 py-2 text-left hover:bg-muted", session.id === currentSessionId && "bg-muted")}><div className="min-w-0 flex-1"><div className="flex items-center gap-1.5"><span className="truncate text-xs font-medium">{session.title}</span>{session.favorite && <Star className="h-3 w-3 shrink-0 fill-amber-400 text-amber-400" />}</div><p className="mt-1 text-[10px] text-muted-foreground">{formatRelativeTime(session.updatedAt)} · {session.messageCount} 条消息</p></div><DropdownMenu.Root><DropdownMenu.Trigger asChild onClick={(event) => event.stopPropagation()}><span className="rounded p-0.5 opacity-0 hover:bg-background group-hover:opacity-100"><MoreHorizontal className="h-3.5 w-3.5" /></span></DropdownMenu.Trigger><DropdownMenu.Portal><DropdownMenu.Content align="end" className="z-50 min-w-36 rounded-md border bg-card p-1 text-xs shadow-lg"><DropdownMenu.Item className="flex cursor-pointer items-center gap-2 rounded px-2 py-1.5 outline-none hover:bg-muted" onSelect={() => void toggleFavorite(session.id, session.favorite)}><Star className="h-3.5 w-3.5" />{session.favorite ? "取消收藏" : "收藏"}</DropdownMenu.Item><DropdownMenu.Item className="flex cursor-pointer items-center gap-2 rounded px-2 py-1.5 text-red-500 outline-none hover:bg-muted" onSelect={() => void deleteSession(session.id)}><Trash2 className="h-3.5 w-3.5" />删除</DropdownMenu.Item></DropdownMenu.Content></DropdownMenu.Portal></DropdownMenu.Root></button></ContextMenu.Trigger><ContextMenu.Portal><ContextMenu.Content className="z-50 min-w-40 rounded-md border bg-card p-1 text-xs shadow-lg"><ContextMenu.Item className="cursor-pointer rounded px-2 py-1.5 outline-none hover:bg-muted" onSelect={() => void toggleFavorite(session.id, session.favorite)}>{session.favorite ? "取消收藏" : "添加到收藏"}</ContextMenu.Item><ContextMenu.Separator className="my-1 h-px bg-border" /><ContextMenu.Item className="cursor-pointer rounded px-2 py-1.5 text-red-500 outline-none hover:bg-muted" onSelect={() => void deleteSession(session.id)}>删除会话</ContextMenu.Item></ContextMenu.Content></ContextMenu.Portal></ContextMenu.Root>)}</div>}
      </>}
    </ScrollArea>
    <div className="border-t p-2"><Tooltip><TooltipTrigger asChild><Button variant="ghost" className={cn("w-full", collapsed ? "px-0" : "justify-start")} onClick={() => navigate("/settings")}><Settings className="h-4 w-4" />{!collapsed && "Settings"}</Button></TooltipTrigger>{collapsed && <TooltipContent side="right">Settings</TooltipContent>}</Tooltip></div>
  </aside>
}
