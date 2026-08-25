import { AlertTriangle, Bot, User } from "lucide-react"
import { memo } from "react"
import { MarkdownRenderer } from "./MarkdownRenderer"
import { Skeleton } from "@/components/ui/skeleton"
import { cn } from "@/lib/utils"
import type { Message } from "@/types/chat"

export const MessageItem = memo(function MessageItem({ message }: { message: Message }) {
  const isUser = message.role === "user"
  const isError = message.role === "error" || message.status === "error"
  return <article className={cn("mx-auto flex w-full max-w-3xl gap-3 px-4 py-5 md:px-8", isUser && "justify-end")}>{!isUser && <div className={cn("mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md border bg-card", isError && "border-red-500/30 text-red-500")}>{isError ? <AlertTriangle className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5" />}</div>}<div className={cn("min-w-0", isUser ? "max-w-[78%] rounded-lg bg-muted px-3.5 py-2.5" : "flex-1")}>
    <div className="mb-1.5 flex items-center gap-2"><span className="text-[11px] font-semibold">{isUser ? "You" : isError ? "Error" : "Agent"}</span><time className="text-[10px] text-muted-foreground">{new Date(message.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time></div>
    {message.status === "streaming" && !message.content ? <div className="space-y-2 py-2"><Skeleton className="h-3 w-4/5" /><Skeleton className="h-3 w-3/5" /></div> : isUser ? <p className="whitespace-pre-wrap text-[13px] leading-5">{message.content}</p> : <MarkdownRenderer content={message.content} />}
  </div>{isUser && <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-foreground text-background"><User className="h-3.5 w-3.5" /></div>}</article>
})
