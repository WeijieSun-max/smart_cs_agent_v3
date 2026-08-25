import { useEffect } from "react"
import { useParams } from "react-router-dom"
import { ChatPanel } from "@/components/chat/ChatPanel"
import { WorkbenchLayout } from "@/components/layout/WorkbenchLayout"
import { useHistory } from "@/hooks/useAgentQueries"
import { useAgentStore } from "@/stores/agentStore"

export function ChatPage() {
  const { sessionId: routeSessionId } = useParams()
  const currentSessionId = useAgentStore((state) => state.currentSessionId)
  const setCurrentSession = useAgentStore((state) => state.setCurrentSession)
  const setMessages = useAgentStore((state) => state.setMessages)
  const sessionId = routeSessionId ?? currentSessionId
  const { data: history } = useHistory(sessionId)
  useEffect(() => { if (routeSessionId && routeSessionId !== currentSessionId) setCurrentSession(routeSessionId) }, [currentSessionId, routeSessionId, setCurrentSession])
  useEffect(() => { if (history) setMessages(history) }, [history, setMessages])
  return <WorkbenchLayout><ChatPanel /></WorkbenchLayout>
}
