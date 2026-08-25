import { AlertTriangle, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return <div className="m-3 rounded-md border border-red-500/20 bg-red-500/5 p-3"><div className="flex gap-2"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-red-500" /><div><p className="text-xs font-medium text-red-600 dark:text-red-400">加载失败</p><p className="mt-1 text-xs text-muted-foreground">{message}</p>{onRetry && <Button variant="outline" size="sm" className="mt-3" onClick={onRetry}><RefreshCw className="h-3 w-3" />重试</Button>}</div></div></div>
}
