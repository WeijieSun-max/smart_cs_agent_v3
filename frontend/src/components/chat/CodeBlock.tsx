import { Check, ChevronDown, ChevronUp, Copy, Maximize2 } from "lucide-react"
import { useState } from "react"
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"
import { oneDark, oneLight } from "react-syntax-highlighter/dist/esm/styles/prism"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog"
import { useUiStore } from "@/stores/uiStore"
import { cn } from "@/lib/utils"

export function CodeBlock({ language = "text", code }: { language?: string; code: string }) {
  const [copied, setCopied] = useState(false)
  const [collapsed, setCollapsed] = useState(false)
  const [expanded, setExpanded] = useState(false)
  const theme = useUiStore((state) => state.theme)
  const copy = async () => { await navigator.clipboard.writeText(code); setCopied(true); window.setTimeout(() => setCopied(false), 1400) }
  const content = <SyntaxHighlighter language={language} style={theme === "dark" ? oneDark : oneLight} customStyle={{ margin: 0, padding: "1rem", background: "transparent", fontSize: "12px", lineHeight: "1.65" }} wrapLongLines>{code.replace(/\n$/, "")}</SyntaxHighlighter>
  return <>
    <div className="my-3 overflow-hidden rounded-md border bg-[#fafafa] dark:bg-[#0d0d0f]">
      <div className="flex h-9 items-center border-b px-3"><span className="font-mono text-[10px] uppercase text-muted-foreground">{language}</span><div className="ml-auto flex items-center gap-0.5"><Button variant="ghost" size="icon-sm" className="h-7 w-7" onClick={() => setCollapsed((value) => !value)} aria-label={collapsed ? "展开代码" : "折叠代码"}>{collapsed ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronUp className="h-3.5 w-3.5" />}</Button><Button variant="ghost" size="icon-sm" className="h-7 w-7" onClick={() => setExpanded(true)} aria-label="全屏查看"><Maximize2 className="h-3.5 w-3.5" /></Button><Button variant="ghost" size="icon-sm" className="h-7 w-7" onClick={() => void copy()} aria-label="复制代码">{copied ? <Check className="h-3.5 w-3.5 text-emerald-500" /> : <Copy className="h-3.5 w-3.5" />}</Button></div></div>
      <div className={cn("max-h-[460px] overflow-auto", collapsed && "hidden")}>{content}</div>
    </div>
    <Dialog open={expanded} onOpenChange={setExpanded}><DialogContent className="flex h-[86vh] max-w-5xl flex-col p-0"><DialogTitle className="border-b px-4 py-3 font-mono text-xs">{language}</DialogTitle><div className="min-h-0 flex-1 overflow-auto">{content}</div></DialogContent></Dialog>
  </>
}
