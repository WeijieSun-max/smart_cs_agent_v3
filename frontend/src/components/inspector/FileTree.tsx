import { ChevronRight, FileCode2, FileJson, FileText, Folder, FolderOpen } from "lucide-react"
import { useState } from "react"
import { cn } from "@/lib/utils"
import type { WorkspaceFile } from "@/types/workspace"

function FileNode({ file, depth = 0 }: { file: WorkspaceFile; depth?: number }) {
  const [open, setOpen] = useState(depth === 0)
  const isFolder = file.type === "folder"
  const FileIcon = file.language === "json" ? FileJson : file.language ? FileCode2 : FileText
  return <div><button onClick={() => isFolder && setOpen((value) => !value)} className="flex h-7 w-full items-center gap-1.5 rounded px-1.5 text-left text-xs hover:bg-muted" style={{ paddingLeft: `${depth * 12 + 6}px` }}>{isFolder ? <><ChevronRight className={cn("h-3 w-3 transition-transform", open && "rotate-90")} />{open ? <FolderOpen className="h-3.5 w-3.5 text-blue-500" /> : <Folder className="h-3.5 w-3.5 text-blue-500" />}</> : <><span className="w-3" /><FileIcon className="h-3.5 w-3.5 text-muted-foreground" /></>}<span className="truncate">{file.name}</span></button>{isFolder && open && file.children?.map((child) => <FileNode key={child.id} file={child} depth={depth + 1} />)}</div>
}

export function FileTree({ files }: { files: WorkspaceFile[] }) { return <div className="space-y-0.5">{files.map((file) => <FileNode key={file.id} file={file} />)}</div> }
