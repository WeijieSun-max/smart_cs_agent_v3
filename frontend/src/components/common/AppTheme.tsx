import { ThemeProvider } from "next-themes"
import type { ReactNode } from "react"
import { useUiStore } from "@/stores/uiStore"

export function AppTheme({ children }: { children: ReactNode }) {
  const theme = useUiStore((state) => state.theme)
  return <ThemeProvider attribute="class" forcedTheme={theme} disableTransitionOnChange>{children}</ThemeProvider>
}
