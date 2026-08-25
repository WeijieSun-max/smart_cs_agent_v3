import { useQueryClient } from "@tanstack/react-query"
import { ArrowLeft, Check, Moon, Server, Sun, UserRound } from "lucide-react"
import { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { queryKeys } from "@/hooks/useAgentQueries"
import { agentService } from "@/services/agent"
import { useAgentStore } from "@/stores/agentStore"
import { useUiStore } from "@/stores/uiStore"

export function SettingsPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const theme = useUiStore((state) => state.theme)
  const toggleTheme = useUiStore((state) => state.toggleTheme)
  const clearContext = useAgentStore((state) => state.clearContext)
  const setCurrentSession = useAgentStore((state) => state.setCurrentSession)
  const [baseUrl, setBaseUrl] = useState(import.meta.env.VITE_API_BASE_URL ?? "/api")
  const [userId, setUserId] = useState("")
  const [savedUserId, setSavedUserId] = useState("")
  const [savingUserId, setSavingUserId] = useState(false)

  useEffect(() => {
    void agentService
      .getCurrentUserId()
      .then((value) => {
        setUserId(value)
        setSavedUserId(value)
      })
      .catch((error: Error) => toast.error(error.message))
  }, [])

  const saveUserId = async () => {
    const value = userId.trim()
    if (!/^[A-Za-z0-9._-]{1,128}$/.test(value)) {
      toast.error("User ID 仅支持字母、数字、点、下划线和连字符")
      return
    }
    setSavingUserId(true)
    try {
      const saved = await agentService.updateCurrentUserId(value)
      setUserId(saved)
      setSavedUserId(saved)
      clearContext()
      setCurrentSession(crypto.randomUUID())
      void queryClient.invalidateQueries({ queryKey: queryKeys.sessions })
      toast.success("当前用户已更新")
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "更新用户失败")
    } finally {
      setSavingUserId(false)
    }
  }

  return (
    <div className="min-h-dvh bg-background">
      <header className="flex h-14 items-center gap-3 border-b px-4">
        <Button variant="ghost" size="icon-sm" onClick={() => navigate(-1)}>
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <div>
          <h1 className="text-sm font-semibold">Settings</h1>
          <p className="text-[10px] text-muted-foreground">Agent Workbench configuration</p>
        </div>
        <Button variant="ghost" size="icon-sm" className="ml-auto" onClick={toggleTheme}>
          {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
      </header>
      <main className="mx-auto max-w-4xl space-y-6 p-4 md:p-8">
        <section className="rounded-lg border bg-card">
          <div className="flex items-center gap-2 border-b px-4 py-3">
            <Server className="h-4 w-4" />
            <h2 className="text-sm font-semibold">API Connection</h2>
            <Badge variant="success" className="ml-auto">
              <Check className="mr-1 h-2.5 w-2.5" />
              Configured
            </Badge>
          </div>
          <div className="grid gap-4 p-4 md:grid-cols-2">
            <label className="space-y-1.5">
              <span className="text-xs font-medium">Base URL</span>
              <input
                value={baseUrl}
                onChange={(event) => setBaseUrl(event.target.value)}
                className="h-9 w-full rounded-md border bg-background px-3 font-mono text-xs outline-none focus:border-primary/50"
              />
            </label>
            <label className="space-y-1.5">
              <span className="text-xs font-medium">Streaming transport</span>
              <select className="h-9 w-full rounded-md border bg-background px-3 text-xs outline-none">
                <option>SSE (recommended)</option>
                <option>WebSocket (reserved)</option>
              </select>
            </label>
          </div>
        </section>
        <section className="rounded-lg border bg-card">
          <div className="flex items-center gap-2 border-b px-4 py-3">
            <UserRound className="h-4 w-4" />
            <div>
              <h2 className="text-sm font-semibold">Current User</h2>
              <p className="mt-0.5 text-[10px] text-muted-foreground">用于唯一标识当前用户</p>
            </div>
          </div>
          <div className="flex flex-col gap-3 p-4 sm:flex-row sm:items-end">
            <label className="flex-1 space-y-1.5">
              <span className="text-xs font-medium">User ID</span>
              <input
                value={userId}
                maxLength={128}
                onChange={(event) => setUserId(event.target.value)}
                placeholder="local-user"
                className="h-9 w-full rounded-md border bg-background px-3 font-mono text-xs outline-none focus:border-primary/50"
              />
            </label>
            <Button
              size="sm"
              disabled={savingUserId || !userId.trim() || userId.trim() === savedUserId}
              onClick={() => void saveUserId()}
            >
              {savingUserId ? "Saving..." : "Save user ID"}
            </Button>
          </div>
        </section>
      </main>
    </div>
  )
}
