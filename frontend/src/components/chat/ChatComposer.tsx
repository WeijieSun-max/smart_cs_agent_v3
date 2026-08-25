import { Eraser, Paperclip, Send, Square } from "lucide-react"
import { useRef, useState } from "react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { QUICK_COMMANDS } from "@/constants/agent"
import { cn } from "@/lib/utils"
import { useAgentStore } from "@/stores/agentStore"

export function ChatComposer({ onSend, onStop }: { onSend: (value: string) => void; onStop: () => void }) {
  const [value, setValue] = useState("")
  const [showCommands, setShowCommands] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const running = useAgentStore((state) => state.agentRunning)
  const clearContext = useAgentStore((state) => state.clearContext)
  const submit = () => { const trimmed = value.trim(); if (!trimmed || running) return; onSend(trimmed); setValue(""); setShowCommands(false) }
  return <div className="relative mx-auto w-full max-w-3xl px-3 pb-3 md:px-6 md:pb-5">
    {showCommands && <div className="absolute bottom-full left-6 right-6 z-20 mb-2 overflow-hidden rounded-md border bg-card p-1 shadow-lg">{QUICK_COMMANDS.map((item) => <button key={item.command} className="flex w-full items-center gap-3 rounded px-2.5 py-2 text-left hover:bg-muted" onClick={() => { setValue(`${item.command} `); setShowCommands(false) }}><code className="w-16 font-mono text-xs text-primary">{item.command}</code><span className="text-xs text-muted-foreground">{item.description}</span></button>)}</div>}
    <div className={cn("rounded-lg border bg-card transition-colors focus-within:border-primary/50", running && "border-primary/30")}>
      <textarea value={value} onChange={(event) => { setValue(event.target.value); setShowCommands(event.target.value === "/") }} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submit() } }} placeholder={running ? "Agent 正在执行任务..." : "向 Agent 发送消息，输入 / 查看快捷指令"} disabled={running} rows={1} className="max-h-40 min-h-[48px] w-full resize-none bg-transparent px-3.5 py-3 text-[13px] leading-5 outline-none placeholder:text-muted-foreground disabled:opacity-60" />
      <div className="flex items-center gap-1 border-t px-2 py-1.5"><input ref={fileRef} type="file" className="hidden" onChange={(event) => event.target.files?.[0] && toast.success(`已附加 ${event.target.files[0].name}`)} /><Button variant="ghost" size="icon-sm" onClick={() => fileRef.current?.click()} aria-label="附加文件"><Paperclip className="h-4 w-4" /></Button><Button variant="ghost" size="icon-sm" onClick={() => { clearContext(); toast.success("上下文已清除") }} aria-label="清除上下文"><Eraser className="h-4 w-4" /></Button><span className="ml-1 text-[10px] text-muted-foreground">Shift + Enter 换行</span><div className="ml-auto">{running ? <Button variant="destructive" size="sm" onClick={onStop}><Square className="h-3 w-3 fill-current" />Stop</Button> : <Button size="sm" disabled={!value.trim()} onClick={submit}><Send className="h-3.5 w-3.5" />Send</Button>}</div></div>
    </div>
  </div>
}
