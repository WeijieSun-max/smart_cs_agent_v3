import { LoaderCircle } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { AGENT_STATUS_LABELS } from "@/constants/agent"
import type { AgentStatus } from "@/types/agent"

export function AgentStatusBadge({ status }: { status: AgentStatus }) {
  const active = ["thinking", "planning", "running_tool", "waiting_tool"].includes(status)
  const variant = status === "failed" ? "destructive" : status === "completed" ? "success" : active ? "default" : "secondary"
  return <Badge variant={variant}>{active && <LoaderCircle className="mr-1 h-2.5 w-2.5 animate-spin" />}{AGENT_STATUS_LABELS[status]}</Badge>
}
