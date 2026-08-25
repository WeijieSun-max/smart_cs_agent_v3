import type { ReactNode } from "react"
import { AppSidebar } from "./AppSidebar"
import { MobileDrawer } from "./MobileDrawer"
import { Inspector } from "@/components/inspector/Inspector"
import { useUiStore } from "@/stores/uiStore"

export function WorkbenchLayout({ children }: { children: ReactNode }) {
  const sidebarMobileOpen = useUiStore((state) => state.sidebarMobileOpen)
  const inspectorMobileOpen = useUiStore((state) => state.inspectorMobileOpen)
  const inspectorVisible = useUiStore((state) => state.inspectorVisible)
  const setSidebarMobileOpen = useUiStore((state) => state.setSidebarMobileOpen)
  const setInspectorMobileOpen = useUiStore((state) => state.setInspectorMobileOpen)
  return <div className="flex h-dvh min-w-0 overflow-hidden bg-background">
    <div className="hidden h-full md:block"><AppSidebar /></div>
    <main className="min-w-0 flex-1">{children}</main>
    {inspectorVisible && <div className="hidden xl:block"><Inspector /></div>}
    <MobileDrawer open={sidebarMobileOpen} onOpenChange={setSidebarMobileOpen} side="left" title="Navigation"><AppSidebar mobile /></MobileDrawer>
    <MobileDrawer open={inspectorMobileOpen} onOpenChange={setInspectorMobileOpen} side="right" title="Inspector"><Inspector mobile /></MobileDrawer>
  </div>
}
