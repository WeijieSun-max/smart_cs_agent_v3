export type MessageRenderStatus = "streaming" | "complete"

interface FrameScheduler {
  request: (callback: FrameRequestCallback) => number
  cancel: (handle: number) => void
}

interface MessageStreamRendererOptions {
  charactersPerFrame?: number
  scheduler?: FrameScheduler
}

export interface MessageStreamRenderer {
  pushDelta: (value: string, cumulative?: boolean) => void
  setFinalAnswer: (value: string) => void
  close: () => void
  cancel: () => void
  finished: Promise<void>
}

const browserScheduler = (): FrameScheduler => ({
  request: (callback) => window.requestAnimationFrame(callback),
  cancel: (handle) => window.cancelAnimationFrame(handle),
})

function splitForFrames(value: string, charactersPerFrame: number): string[] {
  const characters = Array.from(value)
  const chunks: string[] = []
  for (let start = 0; start < characters.length; start += charactersPerFrame) {
    chunks.push(characters.slice(start, start + charactersPerFrame).join(""))
  }
  return chunks
}

export function createMessageStreamRenderer(
  update: (content: string, status: MessageRenderStatus) => void,
  options: MessageStreamRendererOptions = {},
): MessageStreamRenderer {
  const charactersPerFrame = options.charactersPerFrame ?? 6
  if (charactersPerFrame < 1) throw new Error("charactersPerFrame must be positive")

  const scheduler = options.scheduler ?? browserScheduler()
  const queue: string[] = []
  let displayedContent = ""
  let finalAnswer: string | null = null
  let frameHandle: number | null = null
  let closed = false
  let settled = false
  let resolveFinished: () => void = () => undefined
  const finished = new Promise<void>((resolve) => {
    resolveFinished = resolve
  })

  const settle = () => {
    if (settled) return
    settled = true
    resolveFinished()
  }

  function schedule() {
    if (settled || frameHandle !== null) return
    frameHandle = scheduler.request(renderFrame)
  }

  function renderFrame() {
    frameHandle = null
    if (settled) return

    const delta = queue.shift()
    if (delta !== undefined) {
      displayedContent += delta
      update(displayedContent, "streaming")
      if (queue.length > 0 || finalAnswer !== null || closed) schedule()
      return
    }

    if (finalAnswer !== null) {
      displayedContent = finalAnswer
      update(displayedContent, "complete")
      settle()
      return
    }

    if (closed) {
      if (displayedContent) update(displayedContent, "complete")
      settle()
    }
  }

  const pushDelta = (value: string, cumulative = false) => {
    if (settled || !value) return
    const projectedContent = displayedContent + queue.join("")
    const delta =
      cumulative && value.startsWith(projectedContent) ? value.slice(projectedContent.length) : value
    queue.push(...splitForFrames(delta, charactersPerFrame))
    schedule()
  }

  const setFinalAnswer = (value: string) => {
    if (settled) return
    finalAnswer = value
    schedule()
  }

  const close = () => {
    if (settled) return
    closed = true
    schedule()
  }

  const cancel = () => {
    if (frameHandle !== null) scheduler.cancel(frameHandle)
    frameHandle = null
    queue.length = 0
    settle()
  }

  return { pushDelta, setFinalAnswer, close, cancel, finished }
}
