import { Braces, Menu, Moon, PanelRight, Sun } from "lucide-react"
import { Button } from "@/components/ui/button"
import { AgentStatusBadge } from "@/components/agent/AgentStatusBadge"
import { useAgents } from "@/hooks/useAgentQueries"
import { useAgentStore } from "@/stores/agentStore"
import { useUiStore } from "@/stores/uiStore"

export function ChatHeader() {
  const { data: agents } = useAgents()
  const currentAgentId = useAgentStore((state) => state.currentAgentId)
  const status = useAgentStore((state) => state.agentStatus)
  const theme = useUiStore((state) => state.theme)
  const toggleTheme = useUiStore((state) => state.toggleTheme)
  const toggleInspector = useUiStore((state) => state.toggleInspector)
  const setSidebarMobileOpen = useUiStore((state) => state.setSidebarMobileOpen)
  const setInspectorMobileOpen = useUiStore((state) => state.setInspectorMobileOpen)
  const agent = agents?.find((item) => item.id === currentAgentId)
  return <header className="flex h-14 shrink-0 items-center gap-3 border-b bg-background/95 px-3 backdrop-blur md:px-4"><Button variant="ghost" size="icon-sm" className="md:hidden" onClick={() => setSidebarMobileOpen(true)}><Menu className="h-4 w-4" /></Button><div className="min-w-0 flex-1"><div className="flex items-center gap-2"><h1 className="truncate text-sm font-semibold">{agent?.name ?? "Agent"}</h1><AgentStatusBadge status={status} /></div><p className="mt-0.5 truncate text-[10px] text-muted-foreground">{agent?.description ?? "加载 Agent..."} · {agent?.model}</p></div><Button variant="ghost" size="icon-sm" onClick={toggleTheme} aria-label="切换主题">{theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}</Button><Button variant="ghost" size="icon-sm" className="xl:hidden" onClick={() => setInspectorMobileOpen(true)} aria-label="打开 Inspector"><Braces className="h-4 w-4" /></Button><Button variant="ghost" size="icon-sm" className="hidden xl:inline-flex" onClick={toggleInspector} aria-label="切换 Inspector"><PanelRight className="h-4 w-4" /></Button></header>
}
