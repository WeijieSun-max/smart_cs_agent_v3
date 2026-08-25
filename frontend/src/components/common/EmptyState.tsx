import type { LucideIcon } from "lucide-react"
import { MessageSquareDashed } from "lucide-react"

export function EmptyState({ title, description, icon: Icon = MessageSquareDashed }: { title: string; description: string; icon?: LucideIcon }) {
  return <div className="flex h-full min-h-48 flex-col items-center justify-center px-6 text-center"><div className="mb-3 rounded-lg border bg-muted/50 p-3"><Icon className="h-5 w-5 text-muted-foreground" /></div><p className="text-sm font-medium">{title}</p><p className="mt-1 max-w-xs text-xs leading-5 text-muted-foreground">{description}</p></div>
}
