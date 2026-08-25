import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog"

export function MobileDrawer({ open, onOpenChange, side, title, children }: { open: boolean; onOpenChange: (open: boolean) => void; side: "left" | "right"; title: string; children: React.ReactNode }) {
  return <Dialog open={open} onOpenChange={onOpenChange}><DialogContent className={`h-dvh w-[88vw] max-w-[340px] translate-y-0 rounded-none p-0 ${side === "left" ? "left-0 top-0 translate-x-0" : "left-auto right-0 top-0 translate-x-0"}`}><DialogTitle className="sr-only">{title}</DialogTitle>{children}</DialogContent></Dialog>
}
