import { Skeleton } from "@/components/ui/skeleton"

export function LoadingState({ rows = 4 }: { rows?: number }) {
  return <div className="space-y-3 p-3">{Array.from({ length: rows }, (_, index) => <div key={index} className="flex items-center gap-3"><Skeleton className="h-8 w-8 rounded-md" /><div className="flex-1 space-y-2"><Skeleton className="h-3 w-2/3" /><Skeleton className="h-2.5 w-1/2" /></div></div>)}</div>
}
