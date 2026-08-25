import { Check, Copy } from "lucide-react"
import { useMemo, useState } from "react"
import { toast } from "sonner"

function formatValue(value: unknown) {
  if (value === undefined) return "（尚未返回）"
  if (typeof value === "string") return value
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

export function NodeTraceValue({ value }: { value: unknown }) {
  const [copied, setCopied] = useState(false)
  const text = useMemo(() => formatValue(value), [value])
  const redacted = text.includes("[REDACTED_CREDENTIAL]")

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1400)
    } catch {
      toast.error("复制失败，请手动选择内容")
    }
  }

  return <div className="relative">
    <button
      type="button"
      className="absolute right-1.5 top-1.5 z-10 rounded border bg-background/90 p-1 text-muted-foreground shadow-sm hover:text-foreground"
      onClick={() => void copy()}
      aria-label="复制原始数据"
    >
      {copied ? <Check className="h-3 w-3 text-emerald-500" /> : <Copy className="h-3 w-3" />}
    </button>
    <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-all rounded-md border bg-muted/30 py-2 pl-2 pr-8 font-mono text-[10px] leading-4 text-muted-foreground">{text}</pre>
    {redacted && <span className="mt-1 block text-[9px] text-amber-600 dark:text-amber-400">凭据已脱敏</span>}
  </div>
}
