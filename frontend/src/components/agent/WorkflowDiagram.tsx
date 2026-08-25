import mermaid from "mermaid"
import { useEffect, useId, useMemo, useState } from "react"
import { useUiStore } from "@/stores/uiStore"
import type { AgentRunRecord, AgentStep } from "@/types/agent"

function safeLabel(value: string) {
  return value.replace(/["[\]{}<>]/g, " ").replace(/\s+/g, " ").trim().slice(0, 42)
}

function buildDefinition(steps: AgentStep[], runStatus?: AgentRunRecord["status"]) {
  const nodes = steps.map((step, index) => {
    const label = safeLabel(step.title) || `步骤 ${index + 1}`
    const status = step.status === "error" ? "failed" : step.status === "running" ? "running" : "success"
    return `S${index}["${label}"]:::${status}`
  })
  const edges = steps.slice(1).map((_, index) => `S${index} --> S${index + 1}`)
  const terminalClass = runStatus === "failed" || runStatus === "cancelled" ? "failed" : runStatus === "running" ? "running" : "success"
  if (steps.length === 0) nodes.push(`S0["等待执行步骤"]:::${terminalClass}`)
  return `flowchart TD
  ${[...nodes, ...edges].join("\n  ")}
  classDef success fill:#ecfdf5,stroke:#10b981,color:#065f46
  classDef running fill:#eff6ff,stroke:#3b82f6,color:#1e40af
  classDef failed fill:#fef2f2,stroke:#ef4444,color:#991b1b`
}

export default function WorkflowDiagram({ steps, status }: { steps: AgentStep[]; status?: AgentRunRecord["status"] }) {
  const rawId = useId()
  const theme = useUiStore((state) => state.theme)
  const [svg, setSvg] = useState("")
  const definition = useMemo(() => buildDefinition(steps, status), [status, steps])
  useEffect(() => {
    let active = true
    mermaid.initialize({ startOnLoad: false, theme: theme === "dark" ? "dark" : "neutral", securityLevel: "strict", fontFamily: "Inter" })
    void mermaid.render(`workflow-${rawId.replace(/:/g, "")}`, definition).then(({ svg: result }) => {
      if (active) setSvg(result)
    })
    return () => { active = false }
  }, [definition, rawId, theme])
  return <div className="overflow-x-auto rounded-md border bg-background p-2 [&_svg]:mx-auto [&_svg]:max-w-full" dangerouslySetInnerHTML={{ __html: svg }} />
}
