import { lazy, Suspense } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

const CodeBlock = lazy(() => import("./CodeBlock").then((module) => ({ default: module.CodeBlock })))

export function MarkdownRenderer({ content }: { content: string }) {
  return <div className="prose prose-sm max-w-none text-[13px] leading-6 text-foreground dark:prose-invert prose-headings:mb-2 prose-headings:mt-5 prose-headings:text-foreground prose-p:my-2 prose-li:my-0.5 prose-strong:text-foreground prose-a:text-primary">
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
      code({ className, children, ...props }) {
        const match = /language-(\w+)/.exec(className ?? "")
        const code = String(children)
        return match ? <Suspense fallback={<pre className="my-3 overflow-auto rounded-md border bg-muted p-4 font-mono text-xs">{code}</pre>}><CodeBlock language={match[1]} code={code} /></Suspense> : <code className="rounded border bg-muted px-1.5 py-0.5 font-mono text-[11px]" {...props}>{children}</code>
      },
      table: ({ children }) => <div className="my-3 overflow-x-auto"><table className="w-full border-collapse text-xs">{children}</table></div>,
      th: ({ children }) => <th className="border bg-muted px-2 py-1.5 text-left">{children}</th>,
      td: ({ children }) => <td className="border px-2 py-1.5">{children}</td>,
    }}>{content}</ReactMarkdown>
  </div>
}
