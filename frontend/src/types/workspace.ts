export interface WorkspaceFile {
  id: string
  name: string
  path: string
  type: "file" | "folder"
  language?: string
  children?: WorkspaceFile[]
}

export interface ContextItem {
  id: string
  label: string
  type: "system" | "conversation" | "memory" | "retrieval"
  content: string
  tokens: number
}

export interface PlanTask {
  id: string
  title: string
  status: "completed" | "running" | "pending"
}
