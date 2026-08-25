import { lazy, Suspense } from "react"
import { createBrowserRouter, Navigate } from "react-router-dom"
import App from "@/App"
import { ChatPage } from "@/pages/ChatPage"

const SettingsPage = lazy(() => import("@/pages/SettingsPage").then((module) => ({ default: module.SettingsPage })))

export const router = createBrowserRouter([
  { path: "/", element: <App />, children: [
    { index: true, element: <Navigate to="/chat" replace /> },
    { path: "chat", element: <ChatPage /> },
    { path: "chat/:sessionId", element: <ChatPage /> },
    { path: "settings", element: <Suspense fallback={<div className="flex h-dvh items-center justify-center text-sm text-muted-foreground">Loading editor...</div>}><SettingsPage /></Suspense> },
  ] },
])
