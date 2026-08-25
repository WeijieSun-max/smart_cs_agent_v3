import { create } from "zustand"

export type InspectorTab = "context" | "tools" | "tasks" | "debug" | "files"
export type Theme = "light" | "dark"

interface UiState {
  sidebarCollapsed: boolean
  sidebarMobileOpen: boolean
  inspectorVisible: boolean
  inspectorMobileOpen: boolean
  activeInspectorTab: InspectorTab
  theme: Theme
  toggleSidebar: () => void
  setSidebarMobileOpen: (open: boolean) => void
  toggleInspector: () => void
  setInspectorMobileOpen: (open: boolean) => void
  setActiveInspectorTab: (tab: InspectorTab) => void
  toggleTheme: () => void
}

const initialTheme: Theme = localStorage.getItem("agent-theme") === "light" ? "light" : "dark"

export const useUiStore = create<UiState>((set) => ({
  sidebarCollapsed: false,
  sidebarMobileOpen: false,
  inspectorVisible: true,
  inspectorMobileOpen: false,
  activeInspectorTab: "context",
  theme: initialTheme,
  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  setSidebarMobileOpen: (sidebarMobileOpen) => set({ sidebarMobileOpen }),
  toggleInspector: () => set((state) => ({ inspectorVisible: !state.inspectorVisible })),
  setInspectorMobileOpen: (inspectorMobileOpen) => set({ inspectorMobileOpen }),
  setActiveInspectorTab: (activeInspectorTab) => set({ activeInspectorTab }),
  toggleTheme: () => set((state) => {
    const theme = state.theme === "dark" ? "light" : "dark"
    localStorage.setItem("agent-theme", theme)
    return { theme }
  }),
}))
