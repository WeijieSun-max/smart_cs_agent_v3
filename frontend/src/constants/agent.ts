import type { AgentStatus } from "@/types/agent"

export const AGENT_STATUS_LABELS: Record<AgentStatus, string> = {
  idle: "Idle",
  thinking: "Thinking",
  planning: "Planning",
  running_tool: "Running Tool",
  waiting_tool: "Waiting Result",
  completed: "Completed",
  failed: "Failed",
}

export const QUICK_COMMANDS = [
  { command: "/plan", description: "先制定执行计划" },
  { command: "/debug", description: "分析并定位问题" },
  { command: "/search", description: "检索资料与上下文" },
  { command: "/code", description: "编写或修改代码" },
  { command: "/analyze", description: "进行深入分析" },
] as const
